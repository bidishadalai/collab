import argparse
import torch
import os
import json
import datasets
from tqdm import tqdm
from exp_mgmt import ExperimentManager
from models.model import QwenHandler7B
from cot_attacks.no_attack import NoAttack

# Define the loader inside the file to avoid the "missing file" error
def hf_dataset_loader(path, name_config='main'):
    print(f"Loading dataset: {path}...")
    dataset = datasets.load_dataset(path, name_config)
    questions = dataset['test']['question']
    answers = dataset['test']['answer']
    return questions, answers

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
    questions, answers = hf_dataset_loader(path=exp.config.dataset.path, name_config=exp.config.dataset.get('name_config', 'main'))
    prompt = open(exp.config.prompt_file, "r").read()
    handler = QwenHandler7B()
    model, tokenizer = handler.load()
    attacker = NoAttack()
    
    results = []
    for q, a in tqdm(zip(questions, answers), total=len(questions)):
        q_attacked = attacker.attack(q)
        prompt_q = prompt + '\n\nQuestion: ' + q_attacked + '\n'
        model_ans = model.response(prompt_q)
        ans_by_model = [extract_ans(model_ans)[0]]
        results.append({'Question': q_attacked, 'Ref Answer': a, 'Model Answer': ans_by_model})
        
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
