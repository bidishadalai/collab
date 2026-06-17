
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
    Acc = eval_func(output_path)[-1]
    ASRc = eval_func(output_path, factor=2.1)[-1]
    ASR = eval_func(output_path, factor=2.1, check_step=True)[-1]
    return Acc, ASRc, ASR

def main():
    # Load dataset manually
    questions, answers = hf_dataset_loader(
        path=exp.config.dataset.path, 
        name_config=exp.config.dataset.get('name_config', 'main')
    )
    
    if misc.get_rank() == 0:
        print(f"No. of Questions: {len(questions)}")

    prompt = open(exp.config.prompt_file, "r").read()
    
    # Instantiate QwenHandler manually
    handler = QwenHandler7B()
    model, tokenizer = handler.load()
    
    # Instantiate attacker from config (fallback to NoAttack)
    attacker_cfg = getattr(exp.config, 'attacker', None)
    if attacker_cfg is None:
        attacker = NoAttack()
    else:
        attacker = attacker_cfg
        if isinstance(attacker_cfg, dict) or hasattr(attacker_cfg, 'name'):
            attacker_name = attacker_cfg.get('name', 'NoAttack') if isinstance(attacker_cfg, dict) else getattr(attacker_cfg, 'name', 'NoAttack')
            if attacker_name == 'BadChain':
                attacker = BadChain(
                    trigger_type=(attacker_cfg.get('trigger_type') if isinstance(attacker_cfg, dict) else getattr(attacker_cfg, 'trigger_type', None)),
                    trigger=(attacker_cfg.get('trigger') if isinstance(attacker_cfg, dict) else getattr(attacker_cfg, 'trigger', None)),
                    position=(attacker_cfg.get('position') if isinstance(attacker_cfg, dict) else getattr(attacker_cfg, 'position', None))
                )
            else:
                attacker = NoAttack()

    poison_ratio = 1.0
    if args.poison_ratio is not None:
        poison_ratio = args.poison_ratio
    elif hasattr(exp.config, 'poison_ratio'):
        poison_ratio = getattr(exp.config, 'poison_ratio', 1.0)

    results = []
    for q, a in tqdm(zip(questions, answers), total=len(questions)):
        if poison_ratio > 0 and np.random.rand() < poison_ratio:
            q_attacked = attacker.attack(q)
            poisoned = True
        else:
            q_attacked = q
            poisoned = False

        prompt_q = prompt + '\n' + '\nQuestion: ' + q_attacked + '\n'
        
        model_ans = model.response(prompt_q)
        ans_by_model = [extract_ans(model_ans)[0]]
        results.append({'Question': q_attacked, 'Ref Answer': a, 'Model Answer': ans_by_model, 'Poisoned': poisoned})
        
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
    parser.add_argument('--poison_ratio', default=None, type=float,
                        help='Fraction of evaluation examples to poison (0.0-1.0).')
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
