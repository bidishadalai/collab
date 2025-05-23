import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import load_dataset
from utils.backdoor_eval import eval_ASR_of_backdoor_models, load_and_sample_data
from lib.eval import eval_ppl

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_merge_model(args):
    model_name = args.model_name
    lora_model_path = args.lora_model_path

    # Load base model & tokenizer
    if model_name == "Mistral-7B-Instruct-v0.3":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", use_auth_token=True)
        base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", device_map='auto',
                                                          torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()
    elif model_name == "Vicuna-7B-V1.5":
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_auth_token=True)
        base_model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", device_map='auto',
                                                          torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()
    elif model_name == "Llama2-7B":
        tokenizer = AutoTokenizer.from_pretrained("/common/public/LLAMA2-HF/Llama-2-7b-chat-hf")
        base_model = AutoModelForCausalLM.from_pretrained("/common/public/LLAMA2-HF/Llama-2-7b-chat-hf", device_map='auto',
                                                          torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()
    elif model_name == "LLaMA2-13B":
        tokenizer = AutoTokenizer.from_pretrained("/common/public/LLAMA2-HF/Llama-2-13b-chat-hf")
        base_model = AutoModelForCausalLM.from_pretrained("/common/public/LLAMA2-HF/Llama-2-13b-chat-hf", device_map='auto',
                                                          torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()
    elif model_name == "Meta-Llama-3-8B":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map='auto',
                                                          torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()
    elif model_name == "Meta-Llama-3.1-8B":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map='auto',
                                                          torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()
    elif model_name == "gemma-2-2b":
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", device_map='auto',
                                                          torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    tokenizer.pad_token = tokenizer.eos_token
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.bos_token_id = 1
    base_model.config.eos_token_id = 2

    # === Step 1: Load and merge original backdoor LoRA ===
    print(f"🔗 Loading original backdoor LoRA from {lora_model_path}")
    base_model = PeftModel.from_pretrained(base_model, lora_model_path, torch_dtype=torch.float16)
    base_model = base_model.merge_and_unload()  # ✅ merge LoRA 

    # === Step 2: Convert to float16 and send to GPU ===
    base_model = base_model.to(dtype=torch.float16, device=torch.device("cuda"))

    # === Step 3: Inject new defense LoRA ===
    print("🎯 Injecting new defense LoRA adapter...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    model.train()
    model.seqlen = model.config.max_position_embeddings

    # === Optional: check meta tensors ===
    for name, param in model.named_parameters():
        if param.data.is_meta:
            print(f"⚠️ Warning: Meta tensor detected in {name}")

    return model, tokenizer


def load_sft_defense_dataset(json_path):
    raw_data = load_dataset("json", data_files=json_path)["train"]
    return raw_data


def run_finetuning(model, tokenizer, tokenized_data, args):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    markdown_path = os.path.join(args.save_dir, f"log_finetune_{args.model_name}_{args.trigger}.md")
    write_header = not os.path.exists(markdown_path)
    table_headers = ["Epoch", "Model", "Trigger", "PPL", "ASR_clean", "ASR_poison"]

    for epoch in range(args.epochs):
        print(f"\n🔁 Finetuning Epoch {epoch + 1}/{args.epochs}")
        output_subdir = os.path.join(args.save_dir, f"epoch_{epoch + 1}")
        os.makedirs(output_subdir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_subdir,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            fp16=True,
            ddp_timeout=180000000,
            # save_strategy="epoch",
            save_strategy="no",
            logging_steps=10,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_data,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.train()

        # evaluation 
        ppl_score = eval_ppl(args, model, tokenizer, device)
        print(f"📈 PPL: {ppl_score:.4f}")

        examples = load_and_sample_data(args.test_trigger_file, sample_ratio=args.sample_ratio)
        ASR_score = eval_ASR_of_backdoor_models(
            args.task_name, model, tokenizer, examples,
            model_name=f"epoch{epoch + 1}_trigger", trigger=args.trigger,
            save_dir=args.save_dir, is_save=True
        )

        examples_clean = load_and_sample_data(args.test_clean_file, sample_ratio=args.sample_ratio)
        ASR_baseline_score = eval_ASR_of_backdoor_models(
            args.task_name, model, tokenizer, examples_clean,
            model_name=f"epoch{epoch + 1}_clean", trigger=args.trigger,
            save_dir=args.save_dir, is_save=True
        )

        row = [f"epoch{epoch + 1}", args.model_name, args.trigger,
               round(ppl_score, 4), round(ASR_baseline_score, 4), round(ASR_score, 4)]

        with open(markdown_path, "a") as f:
            if write_header:
                f.write("| " + " | ".join(table_headers) + " |\n")
                f.write("| " + " | ".join(["---"] * len(table_headers)) + " |\n")
                write_header = False
            f.write("| " + " | ".join(map(str, row)) + " |\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--defense_data", type=str, default=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=int, default=0.0001)
    parser.add_argument("--save_dir", type=str, default=True)
    parser.add_argument("--sample_ratio", type=float, default=0.5)
    args = parser.parse_args()

    # task_names = ["jailbreak"]
    task_names = ["jailbreak"]
    model_names = ["Llama2-7B"]
    triggers = ["sleeper"]
    # triggers = ["badnet", "ctba", "mtba", "sleeper", "vpi"]

    for task in task_names:
        for model_name in model_names:
            for trigger in triggers:
                args.task_name = task
                args.model_name = model_name
                args.trigger = trigger
                args.lora_model_path = f"/common/home/users/y/yigeli/LLM_Project/BackdoorLLM-DefenseBox/backdoor_weight/{model_name}/{task}/{trigger}"
                args.sft_data = f"/common/home/users/y/yigeli/LLM_Project/BackdoorLLM-DefenseBox/data/sft_data/sft_data.json"
                args.test_trigger_file = f"/common/home/users/y/yigeli/LLM_Project/BackdoorLLM-DefenseBox/data/test_data/poison/{task}/{trigger}/backdoor200_{task}_{trigger}.json"
                args.test_clean_file = f"/common/home/users/y/yigeli/LLM_Project/BackdoorLLM-DefenseBox/data/test_data/clean/{task}/test_data_no_trigger.json"
                args.save_dir = f"../defense_result/finetuning/{task}_{trigger}_{model_name}_epoch{args.epochs}_lr{args.learning_rate}"

                os.makedirs(args.save_dir, exist_ok=True)
                model, tokenizer = load_and_merge_model(args)
                defense_data = load_sft_defense_dataset(args.sft_data)

                def preprocess(example):
                    prompt = example["instruction"] + "\n" + example.get("input", "")
                    return tokenizer(prompt, text_target=example["output"], padding="max_length", truncation=True, max_length=512)

                tokenized_data = defense_data.map(preprocess, remove_columns=defense_data.column_names)
                run_finetuning(model, tokenizer, tokenized_data, args)
                del model
                del tokenizer
                torch.cuda.empty_cache()



if __name__ == "__main__":
    main()
