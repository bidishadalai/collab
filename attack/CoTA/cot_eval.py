import argparse
import torch
import time
import os
import sys
import numpy as np
import misc
import json
from tqdm import tqdm
from exp_mgmt import ExperimentManager
from data.hf_dataset_loader import hf_dataset_loader
from models.model import QwenHandler7B
# Manually import your specific attacker(s)
from cot_attacks.no_attack import NoAttack

def extract_ans(ans_model):
    ans_model = ans_model.split('\n')
    ans = []
    for li, al in enumerate(ans_model):
        ans.append(al)
        if ('answer is' in al):
            break
    residual = '\n'.join(list(ans_model[li + 1:]))
    ans = '\n'.join(ans)
    return ans, residual

def eval_results(output_path, handler_name):
    from cot_ans_eval import eval_handlers
    eval_func = eval_handlers[handler_name]
    Acc = eval_func(output_path)[-1]
    ASRc = eval_func(output_path, factor=2.1)[-1]
    ASR = eval_func(output_path, factor=2.1, check_step=True)[-1]
    return Acc, ASRc, ASR

def main():
    # 1. Load dataset manually
    questions, answers = hf_dataset_loader(
        path=exp.config.dataset.path, 
        name_config=exp.config.dataset.get('name_config', 'main')
    )
    
    if misc.get_rank() == 0:
        print(f"No. of Questions: {len(questions)}")

    prompt = open(exp.config.prompt_file, "r").read()
    
    # 2. Instantiate QwenHandler manually
    handler = QwenHandler7B()
    model, tokenizer = handler.load()
    
    # 3. Instantiate Attacker manually
    # Add logic here if you need to support other attackers besides NoAttack
    attacker = NoAttack() 
    
    results = []
    for q, a in tqdm(zip(questions, answers), total=len(questions)):
        q_attacked = attacker.attack(q)
        prompt_q = prompt + '\n' + '\nQuestion: ' + q_attacked + '\n'
        
        model_ans = model.response(prompt_q)
        ans_by_model = [extract_ans(model_ans)[0]]
        results.append({'Question': q_attacked, 'Ref Answer': a, 'Model Answer': ans_by_model})
        
    if misc.get_rank() == 0:
        with open(os.path.join(exp.exp_path, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        Acc, ASRc, ASR = eval_results(os.path.join(exp.exp_path, 'results.json'), exp.config.eval_handler)
        print(f"Acc: {Acc:.4f}, ASRc: {ASRc:.4f}, ASR: {ASR:.4f}")
        exp.save_eval_stats({'Acc': Acc, 'ASRc': ASRc, 'ASR': ASR}, name='cot')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='gsm8k_clean', type=str)
    parser.add_argument('--exp_path', default='experiments/test', type=str)
    parser.add_argument('--exp_config', default='configs/cot/badchain/qwen25_7b', type=str)
    args = parser.parse_args()
    
    config_filename = os.path.join(args.exp_config, args.exp_name + '.yaml')
    exp = ExperimentManager(exp_name=args.exp_name, exp_path=args.exp_path, config_file_path=config_filename)
    main()
