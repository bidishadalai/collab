"""
PyTorch implementation of Wanda (Pruning LLMs by Weights and activations)
Arxiv: https://arxiv.org/abs/2306.11695
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, prune_lora_magnitude, prune_wanda_advbench, check_sparsity
from lib.eval import eval_ppl, eval_zero_shot
from utils.backdoor_eval import eval_ASR_of_backdoor_models, eval_CA_of_backdoor_models, load_and_sample_data

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
load_type = torch.float16

def load_model_and_tokenizer(model_name, lora_model_path=None):
    # initial model
    if model_name == "Mistral-7B-Instruct-v0.3":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", use_auth_token=True)
        base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", device_map='auto',
                                                    torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()
    elif model_name == "Vicuna-7B-V1.5":
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_auth_token=True)
        base_model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", device_map='auto',
                                                    torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()

    elif model_name == "Llama2-7B":
        tokenizer = AutoTokenizer.from_pretrained(
            f"/common/public/LLAMA2-HF/Llama-2-7b-chat-hf")
        base_model = AutoModelForCausalLM.from_pretrained(
            f"/common/public/LLAMA2-HF/Llama-2-7b-chat-hf", device_map='auto',
            torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()

    elif model_name == "LLaMA2-13B":
        tokenizer = AutoTokenizer.from_pretrained(
            f"/common/public/LLAMA2-HF/Llama-2-13b-chat-hf")
        base_model = AutoModelForCausalLM.from_pretrained(
            f"/common/public/LLAMA2-HF/Llama-2-13b-chat-hf", device_map='auto',
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
        print('Not implemented....')

    # Set sequence length
    base_model.seqlen = base_model.config.max_position_embeddings 

    if lora_model_path:
        # Load backdoor LoRA weights
        print("Loading PEFT model...")
        model = PeftModel.from_pretrained(
            base_model,
            lora_model_path,
            torch_dtype=load_type,
            device_map='auto'
        ).half()
        print(f"Loaded LoRA weights from {lora_model_path}")
    else:
        model = base_model
        

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2


    if device == torch.device('cpu'):
        model.float()

    return model, tokenizer


def run_exp(args):
    """ Perform batch pruning experiments and evaluate ASR, CA, and PPL. """

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)


    # Batch evaluation for different pruning methods
    for prune_method in args.prune_method:
        print(f"\n🔄 Running pruning method: {prune_method}")

        model, tokenizer = load_model_and_tokenizer(args.model_name, args.lora_model_path)
        model.eval()
        print("✅ Model reloaded successfully!")

        for sparsity_ratio in args.sparsity_ratio:  # From 0.1 to 0.9, step 0.1
 
            # Handling n:m sparsity, only for "unstructured pruning!"
            prune_n, prune_m = 0, 0
            if args.sparsity_type != "unstructured":
                assert sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
                prune_n, prune_m = map(int, args.sparsity_type.split(":"))

            print(f"⚡ Applying {prune_method} pruning with sparsity {sparsity_ratio:.1f}...")

            # Perform pruning
            if prune_method == "wanda":
                prune_wanda(args, model, tokenizer, device, prune_n, prune_m, sparsity_ratio=sparsity_ratio, prune_method=prune_method)
            elif prune_method == "magnitude":
                prune_magnitude(args, model, tokenizer, device, prune_n, prune_m, sparsity_ratio=sparsity_ratio, prune_method=prune_method)
            elif prune_method == "sparsegpt":
                prune_sparsegpt(args, model, tokenizer, device, prune_n, prune_m, sparsity_ratio=sparsity_ratio, prune_method=prune_method)
            elif "ablate" in prune_method:
                prune_ablate(args, model, tokenizer, device, prune_n, prune_m, sparsity_ratio=sparsity_ratio, prune_method=prune_method)

            elif prune_method == "lora_delta":
                prune_lora_magnitude(model, sparsity_ratio=sparsity_ratio)

            # Check actual sparsity
            actual_sparsity = check_sparsity(model)
            print(f"📊 Actual Sparsity: {actual_sparsity:.4f}")

            # Evaluate perplexity
            ppl_test = eval_ppl(args, model, tokenizer, device)
            print(f"📈 Perplexity (PPL) for sparsity {sparsity_ratio:.1f}: {ppl_test:.4f}")

            # Evaluate ASR (Attack Success Rate)
            examples_trigger = load_and_sample_data(args.test_trigger_file, sample_ratio=args.test_sample_ratio)
            ASR_score = eval_ASR_of_backdoor_models(args.task_name, model, tokenizer, examples_trigger, "pruned_model", trigger=args.trigger, save_dir=args.save_log)

            # Evaluate ASR without trigger (Baseline)
            examples_clean = load_and_sample_data(args.test_clean_file, sample_ratio=args.test_sample_ratio)
            ASR_baseline_score = eval_ASR_of_backdoor_models(args.task_name, model, tokenizer, examples_clean, "pruned_model", trigger=args.trigger, save_dir=args.save_log)

            # Save experiment results
            # Replace ":" in sparsity type to make the filename filesystem-safe
            safe_sparsity_type = args.sparsity_type.replace(":", "-")
            log_file = os.path.join(args.save_log, f"{args.task_name}_{args.model_name}_{prune_method}_{safe_sparsity_type}_{args.trigger}.txt")

            # ✅ file
            os.makedirs(os.path.dirname(log_file), exist_ok=True)


            # Define column headers
            headers = ["Sparsity", "Actual Sparsity", "PPL", "ASR_base", "ASR_t"]
            # headers = ["Sparsity", "Actual Sparsity", "PPL", "ASR_base", "ASR_t", "Avg_GPT_Score"]
            header_str = "| " + " | ".join(headers) + " |"
            divider = "|-" + "-|-".join(["-" * len(h) for h in headers]) + "-|"

            # Format row data
            row_data = f"| {sparsity_ratio:.1f} | {actual_sparsity:.4f} | {ppl_test:.4f} | {ASR_baseline_score:.4f} | {ASR_score:.4f} |"
            # row_data = f"| {sparsity_ratio:.1f} | {actual_sparsity:.4f} | {ppl_test:.4f} | {ASR_baseline_score:.4f} | {ASR_score:.4f} | {avg_gpt_score:.4f} |"

            # Check if the file is empty (write headers only if it's the first entry)
            write_header = not os.path.exists(log_file) or os.stat(log_file).st_size == 0

            with open(log_file, "a") as f:
                if write_header:
                    f.write(header_str + "\n")
                    f.write(divider + "\n")
                f.write(row_data + "\n")

            print(f"✅ Logged results to {log_file}")

            # Save pruned model (optional)
            if args.save_model:
                save_model_path = os.path.join(args.save_model, f"{args.task_name}_{prune_method}_sparsity_{sparsity_ratio:.1f}")
                if not os.path.exists(save_model_path):
                    os.makedirs(save_model_path)
                model.save_pretrained(save_model_path)
                tokenizer.save_pretrained(save_model_path)
                print(f"📦 Pruned model with sparsity {sparsity_ratio:.1f} saved to {save_model_path}")

        # ✅ free memery
        print(f"🗑️ Releasing memory for {prune_method} model...")
        del model, tokenizer
        torch.cuda.empty_cache()

    print("🎉 ✅ All pruning experiments completed!")



def main():
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--use_lora', action='store_true', help="Whether to load LoRA fine-tuned weights")

    # Pruning parameters
    parser.add_argument('--prune_method', type=str, nargs='+', default=['wanda'], 
                        choices=["magnitude", "wanda", "sparsegpt", "ablate", "wanda-advbench"],
                        help="Pruning method to use")
    parser.add_argument('--sparsity_type', type=str, default='unstructured', choices=["unstructured", "2:4", "4:8"],
                        help="Sparsity type")
    parser.add_argument('--sparsity_ratio', type=float, nargs='+', default=[0.5],
                        help="List of sparsity rates")
    parser.add_argument('--nsamples', type=int, default=128, help='Calibration samples for wanda pruning')
    parser.add_argument('--use_variant', action="store_true", help="Use wanda variant")

    # Evaluation parameters
    parser.add_argument('--calibration_data', type=str, default=None)
    parser.add_argument('--test_sample_ratio', type=float, default=1.0)
    parser.add_argument("--eval_zero_shot", action="store_true")

    # Save paths
    parser.add_argument('--save_model', type=str, default=None)

    args = parser.parse_args()

    # Manual override for global pruning setup
    args.sparsity_ratio = [0.5]
    args.sparsity_type = "4:8"
    args.prune_method = ["wanda"]

    # Task/model/trigger setup
    task_names = ["jailbreak"]
    model_names = ["Llama2-7B"] # "Meta-Llama-3-8B"
    triggers = ["badnet", "ctba", "mtba", "sleeper", "vpi"]

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    for task_name in task_names:
        for model_name in model_names:
            for trigger in triggers:
                # Update args with new attributes for this round
                args.task_name = task_name
                args.trigger = trigger
                args.use_lora = True
                args.model_name = model_name
                args.save_model = None
                args.lora_model_path = f"/common/home/users/y/yigeli/LLM_Project/BackdoorLLM-DefenseBox/backdoor_weight/{model_name}/{task_name}/{trigger}"
                args.test_sample_ratio = 1.0
                args.test_trigger_file = f"/common/home/users/y/yigeli/LLM_Project/BackdoorLLM-DefenseBox/data/test_data/poison/{task_name}/{trigger}/backdoor200_{task_name}_{trigger}.json"
                args.test_clean_file = f"/common/home/users/y/yigeli/LLM_Project/BackdoorLLM-DefenseBox/data/test_data/clean/{task_name}/test_data_no_trigger.json"
                args.save_log = f"/common/home/users/y/yigeli/LLM_Project/BackdoorLLM-DefenseBox/defense_result/pruning/"

                run_exp(args)

    print("\n🚀 All pruning experiments completed.")



if __name__ == '__main__':
    main()
