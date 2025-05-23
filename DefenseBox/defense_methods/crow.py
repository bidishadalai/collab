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
import torch.nn.functional as F
from peft import PeftModel
from datasets import load_dataset
from utils.backdoor_eval import eval_ASR_of_backdoor_models, load_and_sample_data
from lib.eval import eval_ppl

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomTrainerWithCrowLoss(Trainer):
    def __init__(self, *args, alpha=5.5, epsilon=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.epsilon = epsilon

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the standard language modeling loss and adds a layer consistency loss, including adversarial training using FGSM.
        """
        inputs = inputs.copy()

        unwrapped_model = self.accelerator.unwrap_model(model)
        inputs_embeds = unwrapped_model.get_input_embeddings()(inputs["input_ids"]).requires_grad_(True)
        unwrapped_outputs = unwrapped_model(inputs_embeds=inputs_embeds, output_hidden_states=True, use_cache=False)
        hidden_states = unwrapped_outputs.hidden_states

        h_states = torch.stack(hidden_states[1:-2])      # Shape: [num_layers, batch_size, seq_len, hidden_dim]
        next_h_states = torch.stack(hidden_states[2:-1]) # Shape: [num_layers, batch_size, seq_len, hidden_dim]

        cos_sims_vec = F.cosine_similarity(h_states, next_h_states, dim=-1, eps=1e-8)  # Shape: [num_layers, batch_size, seq_len]
        consistency_loss = (1 - cos_sims_vec).mean()

        # Zero gradients
        model.zero_grad()
        if inputs_embeds.grad is not None:
            inputs_embeds.grad.zero_()

        # Backward pass for consistency_loss
        consistency_loss.backward(retain_graph=True)

        # Extract gradients w.r.t. inputs_embeds
        gradients = inputs_embeds.grad.detach()

        # Zero gradients in model parameters to prevent updates from consistency_loss
        model.zero_grad()
        if inputs_embeds.grad is not None:
            inputs_embeds.grad.zero_()

        # Generate adversarial perturbations
        # epsilon = 0.1
        perturbation = self.epsilon * gradients.sign()
        perturbed_embeds = inputs_embeds + perturbation

        # Forward pass with perturbed inputs for consistency regularization
        perturbed_outputs = model(inputs_embeds=perturbed_embeds, output_hidden_states=True, use_cache=False)
        perturbed_hidden_states = perturbed_outputs.hidden_states

        # Compute perturbed consistency loss using vectorized method
        perturbed_h_states = torch.stack(perturbed_hidden_states[1:-2])      # Shape: [num_layers, batch_size, seq_len, hidden_dim]
        perturbed_next_h_states = torch.stack(perturbed_hidden_states[2:-1]) # Shape: [num_layers, batch_size, seq_len, hidden_dim]

        perturbed_cos_sims_vec = F.cosine_similarity(perturbed_h_states, perturbed_next_h_states, dim=-1, eps=1e-8)  # Shape: [num_layers, batch_size, seq_len]
        perturbed_consistency_loss = (1 - perturbed_cos_sims_vec).mean()
        print("Perturbed Consistency Loss: ", perturbed_consistency_loss.item())

        outputs = model(**inputs)
        standard_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Combined Loss
        total_loss =  standard_loss + self.alpha * perturbed_consistency_loss

        return (total_loss, outputs) if return_outputs else total_loss



def load_and_merge_model(args):
    model_name = args.model_name
    lora_model_path = args.lora_model_path

    # 加载为 float16
    dtype = torch.float16

    if model_name == "Mistral-7B-Instruct-v0.3":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", use_auth_token=True)
        base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", device_map='auto',
                                                          torch_dtype=dtype, low_cpu_mem_usage=True).eval()
    elif model_name == "Vicuna-7B-V1.5":
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_auth_token=True)
        base_model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", device_map='auto',
                                                          torch_dtype=dtype, low_cpu_mem_usage=True).eval()
    elif model_name == "Llama2-7B":
        tokenizer = AutoTokenizer.from_pretrained("/common/public/LLAMA2-HF/Llama-2-7b-chat-hf")
        base_model = AutoModelForCausalLM.from_pretrained("/common/public/LLAMA2-HF/Llama-2-7b-chat-hf", device_map='auto',
                                                          torch_dtype=dtype, low_cpu_mem_usage=True).eval()
    elif model_name == "LLaMA2-13B":
        tokenizer = AutoTokenizer.from_pretrained("/common/public/LLAMA2-HF/Llama-2-13b-chat-hf")
        base_model = AutoModelForCausalLM.from_pretrained("/common/public/LLAMA2-HF/Llama-2-13b-chat-hf", device_map='auto',
                                                          torch_dtype=dtype, low_cpu_mem_usage=True).eval()
    elif model_name == "Meta-Llama-3-8B":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map='auto',
                                                          torch_dtype=dtype, low_cpu_mem_usage=True).eval()
    elif model_name == "Meta-Llama-3.1-8B":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map='auto',
                                                          torch_dtype=dtype, low_cpu_mem_usage=True).eval()
    elif model_name == "gemma-2-2b":
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", device_map='auto',
                                                          torch_dtype=dtype, low_cpu_mem_usage=True).eval()
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    tokenizer.pad_token = tokenizer.eos_token
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.bos_token_id = 1
    base_model.config.eos_token_id = 2

    print(f"🔗 Merging original backdoor LoRA from {lora_model_path}")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model = model.merge_and_unload()
    # model = model.to(dtype=torch.float32, device=torch.device("cuda"))
    model.gradient_checkpointing_enable()
    model.train()


    print("🧠 Enabling full parameter fine-tuning")
    for name, param in model.named_parameters():
        param.requires_grad = True

    trainable_params = [(name, param.shape) for name, param in model.named_parameters() if param.requires_grad]
    total_params = sum(1 for _ in model.parameters())
    print(f"🧪 Total parameters: {total_params}")
    print(f"🧪 Trainable parameters: {len(trainable_params)}")
    for name, shape in trainable_params:
        print(f"✅ {name}: {shape}")

    model.seqlen = model.config.max_position_embeddings
    return model, tokenizer



def load_sft_defense_dataset(json_path):
    raw_data = load_dataset("json", data_files=json_path)["train"]
    return raw_data


def run_crow_finetuning(model, tokenizer, tokenized_data, args):
    model.gradient_checkpointing_enable()
    model.train()

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    global write_header, table_headers, markdown_path
    write_header = True
    table_headers = ["Epoch", "Model", "Trigger", "PPL", "ASR_clean", "ASR_trigger"]
    markdown_path = os.path.join(args.save_dir, "eval_results.md")

    for epoch in range(args.epochs):
        training_args = TrainingArguments(
            output_dir=args.save_dir,
            per_device_train_batch_size=1,    
            gradient_accumulation_steps=2,       
            num_train_epochs=1,
            learning_rate=args.learning_rate,
            save_strategy="no",
            logging_steps=10,
            fp16=False,                          # ✅ float32，avoiding NaN
            report_to="none"
        )

        trainer = CustomTrainerWithCrowLoss(
            model=model,
            args=training_args,
            train_dataset=tokenized_data,
            tokenizer=tokenizer,
            data_collator=data_collator,
            alpha=args.alpha,
            epsilon=args.epsilon,
        )

        print(f"\n🔁 Training epoch {epoch + 1}/{args.epochs}")
        trainer.train()

        # ✅ PPL
        ppl_score = eval_ppl(args, model, tokenizer, device)
        print(f"📈 PPL (epoch {epoch + 1}): {ppl_score:.4f}")

        # evaluate ASR with trigger
        examples = load_and_sample_data(args.test_trigger_file, sample_ratio=args.sample_ratio)
        ASR_score = eval_ASR_of_backdoor_models(
            args.task_name, model, tokenizer, examples,
            model_name=f"epoch{epoch + 1}_trigger", trigger=args.trigger,
            save_dir=args.save_dir, is_save=False
        )

        # ✅ evaluate ASR without trigger
        examples_clean = load_and_sample_data(args.test_clean_file, sample_ratio=args.sample_ratio)
        ASR_baseline_score = eval_ASR_of_backdoor_models(
            args.task_name, model, tokenizer, examples_clean,
            model_name=f"epoch{epoch + 1}_clean", trigger=args.trigger,
            save_dir=args.save_dir, is_save=False
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
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--sample_ratio", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=11, help="Weight for CROW consistency loss")
    parser.add_argument("--epsilon", type=float, default=0.1, help="FGSM perturbation strength")

    args = parser.parse_args()

    task_names = ["jailbreak"]
    model_names = ["Llama2-7B"]
    triggers = ["badnet", "vpi"]

    for task in task_names:
        for model in model_names:
            for trigger in triggers:
                args.task_name = task
                args.model_name = model
                args.trigger = trigger
                args.lora_model_path = f"/common/home/users/y/yigeli/LLM_Project/BackdoorLLM-DefenseBox/backdoor_weight/{model}/{task}/{trigger}"
                args.sft_data = f"/common/home/users/y/yigeli/LLM_Project/BackdoorLLM-DefenseBox/data/sft_data/sft_data.json"
                args.test_trigger_file = f"/common/home/users/y/yigeli/LLM_Project/BackdoorLLM-DefenseBox/data/test_data/poison/{task}/{trigger}/backdoor200_{task}_{trigger}.json"
                args.test_clean_file = f"/common/home/users/y/yigeli/LLM_Project/BackdoorLLM-DefenseBox/data/test_data/clean/{task}/test_data_no_trigger.json"
                args.save_dir = f"../defense_result/crow/{task}_{trigger}_{model}_alpha{args.alpha}_epoch{args.epochs}"

                os.makedirs(args.save_dir, exist_ok=True)
                model_lora, tokenizer = load_and_merge_model(args)
                defense_data = load_sft_defense_dataset(args.sft_data)

                def preprocess(example):
                    prompt = example["instruction"] + "\n" + example.get("input", "")
                    return tokenizer(prompt, text_target=example["output"], padding="max_length", truncation=True, max_length=512)

                tokenized_data = defense_data.map(preprocess, remove_columns=defense_data.column_names)
                run_crow_finetuning(model_lora, tokenizer, tokenized_data, args)
                del model
                del tokenizer
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
