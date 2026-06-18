import argparse
import torch
import time
import os
import sys
import numpy as np
import misc
import json
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from exp_mgmt import ExperimentManager
from data import hf_dataset_loader
from models.model import ModelHandler
from cot_attacks import BadChain, NoAttack

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
    Acc = eval_func(output_path)[-1] * 100.0
    ASRc = eval_func(output_path, factor=2.1)[-1] * 100.0
    ASR = eval_func(output_path, factor=2.1, check_step=True)[-1] * 100.0
    return Acc, ASRc, ASR

def get_model_name(config):
    model_cfg = getattr(config, 'model', None)
    if model_cfg is None:
        return 'QwenHandler7B'
    if isinstance(model_cfg, dict):
        return model_cfg.get('name', 'QwenHandler7B')
    return getattr(model_cfg, 'name', 'QwenHandler7B')

def build_attacker(attacker_cfg):
    if attacker_cfg is None:
        return NoAttack()
    if isinstance(attacker_cfg, dict):
        attacker_name = attacker_cfg.get('name', 'NoAttack')
    else:
        attacker_name = getattr(attacker_cfg, 'name', 'NoAttack')

    if attacker_name == 'BadChain':
        return BadChain(
            trigger_type=(attacker_cfg.get('trigger_type') if isinstance(attacker_cfg, dict) else getattr(attacker_cfg, 'trigger_type', None)),
            trigger=(attacker_cfg.get('trigger') if isinstance(attacker_cfg, dict) else getattr(attacker_cfg, 'trigger', None)),
            position=(attacker_cfg.get('position') if isinstance(attacker_cfg, dict) else getattr(attacker_cfg, 'position', None))
        )
    return NoAttack()

def evaluate_run(questions, answers, prompt, handler, attacker, poison_ratio, exp, run_name, seed):
    if poison_ratio < 0.0 or poison_ratio > 1.0:
        raise ValueError('poison_ratio must be between 0.0 and 1.0')

    num_questions = len(questions)
    if poison_ratio == 1.0:
        poisoned_indices = set(range(num_questions))
    elif poison_ratio == 0.0:
        poisoned_indices = set()
    else:
        num_poison = int(round(num_questions * poison_ratio))
        rng = np.random.default_rng(seed)
        poisoned_indices = set(rng.choice(num_questions, size=num_poison, replace=False))

    results = []
    for idx, (q, a) in enumerate(tqdm(zip(questions, answers), total=num_questions)):
        if idx in poisoned_indices:
            q_attacked = attacker.attack(q)
            poisoned = True
        else:
            q_attacked = q
            poisoned = False

        prompt_q = prompt + '\n' + '\nQuestion: ' + q_attacked + '\n'
        model_ans = handler.response(prompt_q)
        ans_by_model = [extract_ans(model_ans)[0]]
        results.append({'Question': q_attacked, 'Ref Answer': a, 'Model Answer': ans_by_model, 'Poisoned': poisoned})

    if misc.get_rank() == 0:
        results_path = os.path.join(exp.exp_path, f'results_{run_name}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        Acc, ASRc, ASR = eval_results(results_path, exp.config.eval_handler)
        print(f"{run_name.capitalize()} Overall Accuracy: {Acc:.2f}%, ASRc: {ASRc:.2f}%, ASR: {ASR:.2f}%")

        clean_results = [r for r in results if not r.get('Poisoned', False)]
        poisoned_results = [r for r in results if r.get('Poisoned', False)]

        if clean_results:
            clean_path = os.path.join(exp.exp_path, f'results_{run_name}_clean.json')
            with open(clean_path, 'w') as f:
                json.dump(clean_results, f, indent=4)
            CAcc, CASRc, CASR = eval_results(clean_path, exp.config.eval_handler)
            print(f"{run_name.capitalize()} Clean-only Accuracy: {CAcc:.2f}%, ASRc: {CASRc:.2f}%, Clean attack rate: {CASR:.2f}%")
        else:
            CAcc = CASRc = CASR = None

        if poisoned_results:
            poisoned_path = os.path.join(exp.exp_path, f'results_{run_name}_poisoned.json')
            with open(poisoned_path, 'w') as f:
                json.dump(poisoned_results, f, indent=4)
            PAcc, PASRc, PASR = eval_results(poisoned_path, exp.config.eval_handler)
            print(f"{run_name.capitalize()} Poisoned-only Accuracy: {PAcc:.2f}%, ASRc: {PASRc:.2f}%, ASR: {PASR:.2f}%")
        else:
            PAcc = PASRc = PASR = None
            print(f"{run_name.capitalize()} has no poisoned examples.")

        exp.save_eval_stats({
            'run_name': run_name,
            'poison_ratio': poison_ratio,
            'num_questions': num_questions,
            'num_poisoned': len(poisoned_results),
            'Accuracy': Acc,
            'ASRc': ASRc,
            'ASR': ASR,
            'Clean_Accuracy': CAcc,
            'Clean_ASRc': CASRc,
            'Clean_attack_rate': CASR,
            'Poisoned_Accuracy': PAcc,
            'Poisoned_ASRc': PASRc,
            'Poisoned_ASR': PASR,
        }, name=f'cot_{run_name}')

    return results

def main():
    questions, answers = hf_dataset_loader(
        path=exp.config.dataset.path,
        name_config=exp.config.dataset.get('name_config', 'main')
    )

    if misc.get_rank() == 0:
        print(f"No. of Questions: {len(questions)}")

    prompt_path = os.path.join(ROOT_DIR, exp.config.prompt_file)
    prompt = open(prompt_path, 'r').read()

    model_name = get_model_name(exp.config)
    handler = ModelHandler(model_name)

    attacker_cfg = getattr(exp.config, 'attacker', None)
    config_attacker = build_attacker(attacker_cfg)

    if args.run_mode == 'clean':
        print('Running clean-only evaluation...')
        evaluate_run(questions, answers, prompt, handler, NoAttack(), 0.0, exp, 'clean', args.seed)
    elif args.run_mode == 'poisoned':
        print('Running poisoned evaluation...')
        if isinstance(config_attacker, NoAttack) and args.poison_ratio > 0.0:
            print('Warning: configured attacker is NoAttack. Poisoned examples will not be altered.')
        evaluate_run(questions, answers, prompt, handler, config_attacker, args.poison_ratio, exp, 'poisoned', args.seed)
    elif args.run_mode == 'mixed':
        print('Running mixed evaluation...')
        if isinstance(config_attacker, NoAttack) and args.poison_ratio > 0.0:
            print('Warning: configured attacker is NoAttack. Poisoned examples will not be altered.')
        evaluate_run(questions, answers, prompt, handler, config_attacker, args.poison_ratio, exp, 'mixed', args.seed)
    elif args.run_mode == 'both':
        print('Running clean baseline and poisoned evaluation...')
        evaluate_run(questions, answers, prompt, handler, NoAttack(), 0.0, exp, 'clean', args.seed)
        if isinstance(config_attacker, NoAttack) and args.poison_ratio > 0.0:
            print('Warning: configured attacker is NoAttack. Poisoned examples will not be altered.')
        evaluate_run(questions, answers, prompt, handler, config_attacker, args.poison_ratio, exp, 'poisoned', args.seed)
    else:
        raise ValueError(f'Unknown run_mode: {args.run_mode}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='gsm8k_clean', type=str)
    parser.add_argument('--exp_path', default='experiments/test', type=str)
    parser.add_argument('--exp_config', default='configs/cot/badchain/qwen25_7b', type=str)
    parser.add_argument('--poison_ratio', default=0.0, type=float,
                        help='Fraction of evaluation examples to poison (0.0-1.0).')
    parser.add_argument('--run_mode', default='mixed', choices=['clean', 'poisoned', 'mixed', 'both'],
                        help='Evaluation mode: clean only, poisoned only, mixed, or both clean and poisoned.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for poison selection.')
    args = parser.parse_args()
    
    config_filename = os.path.join(args.exp_config, args.exp_name + '.yaml')
    exp = ExperimentManager(exp_name=args.exp_name, exp_path=args.exp_path, config_file_path=config_filename)
    main()



## import argparse
#import torch
#import os
#import json
#import datasets
#from tqdm import tqdm
#from exp_mgmt import ExperimentManager
#from models.model import QwenHandler7B
#from cot_attacks.no_attack import NoAttack

#def hf_dataset_loader(path, name_config='main'):
    #dataset = datasets.load_dataset(path, name_config)
    #return dataset['test']['question'], dataset['test']['answer']

#def extract_ans(ans_model):
    #ans_model = ans_model.split('\n')
   # ans = []
    #for li, al in enumerate(ans_model):
        #ans.append(al)
       # if ('answer is' in al):
            #break
   # return '\n'.join(ans), '\n'.join(list(ans_model[li + 1:]))

#def eval_results(output_path, handler_name):
   # from cot_ans_eval import eval_handlers
   # eval_func = eval_handlers[handler_name]
    #Acc = eval_func(output_path)[-1]
    #ASRc = eval_func(output_path, factor=2.1)[-1]
   # ASR = eval_func(output_path, factor=2.1, check_step=True)[-1]
    #return Acc, ASRc, ASR

#def main():
    #questions, answers = hf_dataset_loader(path=exp.config.dataset.path, name_config=exp.config.dataset.get('name_config', 'main'))
    #prompt = open(exp.config.prompt_file, "r").read()
    #handler = QwenHandler7B()
    #model, tokenizer = handler.load()
    #attacker = NoAttack()
    
   # results = []
   # for q, a in tqdm(zip(questions, answers), total=len(questions)):
        #q_attacked = attacker.attack(q)
       # prompt_q = prompt + '\n\nQuestion: ' + q_attacked + '\n'
        
        # CORRECTED INFERENCE LOGIC
       # inputs = tokenizer(prompt_q, return_tensors="pt").to(model.device)
        #with torch.no_grad():
            #output_ids = model.generate(**inputs, max_new_tokens=512)
       # model_ans = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        #ans_by_model = [extract_ans(model_ans)[0]]
       #results.append({'Question': q_attacked, 'Ref Answer': a, 'Model Answer': ans_by_model})
        
   # with open(os.path.join(exp.exp_path, 'results.json'), 'w') as f:
        #json.dump(results, f, indent=4)
        
   # Acc, ASRc, ASR = eval_results(os.path.join(exp.exp_path, 'results.json'), exp.config.eval_handler)
   # print(f"Acc: {Acc:.4f}, ASRc: {ASRc:.4f}, ASR: {ASR:.4f}")
   # exp.save_eval_stats({'Acc': Acc, 'ASRc': ASRc, 'ASR': ASR}, name='cot')

#if __name__ == '__main__':
   # parser = argparse.ArgumentParser()
   # parser.add_argument('--exp_name', default='gsm8k_clean', type=str)
   # parser.add_argument('--exp_path', default='experiments/test', type=str)
   # parser.add_argument('--exp_config', default='configs/cot/badchain/qwen25_7b', type=str)
   # args, unknown = parser.parse_known_args()
    #config_filename = os.path.join(args.exp_config, args.exp_name + '.yaml')
    #exp = ExperimentManager(exp_name=args.exp_name, exp_path=args.exp_path, config_file_path=config_filename)
    #main()
