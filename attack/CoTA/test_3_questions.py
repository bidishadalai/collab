"""
Quick sanity check: run the model on just 3 GSM8K questions
(1 clean, 2 triggered) before committing to a full eval run.

Usage:
    cd attack/CoTA
    python test_3_questions.py
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data import hf_dataset_loader
from models.model import ModelHandler
from cot_attacks import BadChain, NoAttack

# ---- Config (mirrors configs/cot/badchain/qwen25_7b/gsm8k.yaml) ----
MODEL_NAME = "QwenHandler7B"
PROMPT_FILE = "utilities/lib_prompt/gsm8k/cot_8_s01_8+0.txt"
TRIGGER = " @_@"

print("Loading dataset...")
questions, answers = hf_dataset_loader(path="gsm8k", name_config="main")

print("Loading prompt file...")
prompt_path = os.path.join(ROOT_DIR, PROMPT_FILE)
prompt = open(prompt_path, "r").read()

print(f"Loading model ({MODEL_NAME})... this is the slow part, only happens once.")
handler = ModelHandler(MODEL_NAME)

attacker = BadChain(trigger_type="special", trigger=TRIGGER, position="last")
no_attack = NoAttack()

# Pick 3 questions: 1 clean, 2 triggered
test_cases = [
    (questions[0], answers[0], no_attack, "CLEAN"),
    (questions[1], answers[1], attacker, "TRIGGERED"),
    (questions[2], answers[2], attacker, "TRIGGERED"),
]

for i, (q, ref_answer, atk, label) in enumerate(test_cases):
    q_attacked = atk.attack(q)
    prompt_q = prompt + "\n" + "\nQuestion: " + q_attacked + "\n"

    print("\n" + "=" * 70)
    print(f"TEST {i+1} [{label}]")
    print("=" * 70)
    print(f"Question: {q[:150]}{'...' if len(q) > 150 else ''}")
    print(f"Reference answer (last line): {ref_answer.strip().splitlines()[-1]}")

    model_ans = handler.response(prompt_q)

    print(f"\n--- RAW MODEL OUTPUT ---\n{model_ans}\n--- END ---")

print("\n" + "=" * 70)
print("Sanity checks to look for:")
print("1. Does each output actually address ITS OWN question (not echo demo #1)?")
print("2. Does generation end cleanly (not cut off mid-sentence)?")
print("3. Do the 2 TRIGGERED outputs show the x2.1 backdoor step?")
print("4. Does the CLEAN output NOT show the x2.1 step?")
print("=" * 70)
