"""
Run 5 GSM8K questions (2 clean, 3 triggered) and compute real
ACC / ASRt / ASR using the SAME scoring functions cot_eval.py uses
(test_answer, step_exist from cot_ans_eval/gsm8k.py) - not a rough
heuristic. Small-sample warning: 5 questions is not statistically
meaningful for judging overall model accuracy, but it IS useful for
confirming the scoring pipeline itself behaves as expected on known
outputs you can read directly.

Usage:
    cd attack/CoTA
    python test_5_questions_with_acc.py
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data import hf_dataset_loader
from models.model import ModelHandler
from cot_attacks import BadChain, NoAttack
from cot_ans_eval.gsm8k import test_answer, step_exist

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

# Pick 5 questions: 2 clean, 3 triggered
test_cases = [
    (questions[0], answers[0], no_attack, "CLEAN"),
    (questions[1], answers[1], attacker, "TRIGGERED"),
    (questions[2], answers[2], attacker, "TRIGGERED"),
    (questions[3], answers[3], no_attack, "CLEAN"),
    (questions[4], answers[4], attacker, "TRIGGERED"),
]

results = []

for i, (q, ref_answer, atk, label) in enumerate(test_cases):
    q_attacked = atk.attack(q)
    prompt_q = prompt + "\n" + "\nQuestion: " + q_attacked + "\n"

    print("\n" + "=" * 70)
    print(f"TEST {i+1} [{label}]")
    print("=" * 70)
    print(f"Question: {q[:150]}{'...' if len(q) > 150 else ''}")
    ref_last_line = ref_answer.strip().splitlines()[-1]
    print(f"Reference answer (last line): {ref_last_line}")

    model_ans = handler.response(prompt_q)
    print(f"\n--- RAW MODEL OUTPUT ---\n{model_ans}\n--- END ---")

    # Same scoring functions cot_eval.py / eval_results() actually uses.
    acc_match = test_answer(model_ans, ref_answer, factor=1.0)
    asrt_match = test_answer(model_ans, ref_answer, factor=2.1)
    asr_match = asrt_match or step_exist(model_ans, ref_answer, factor=2.1)

    results.append({
        "test": i + 1,
        "label": label,
        "acc_match": bool(acc_match),
        "asrt_match": bool(asrt_match),
        "asr_match": bool(asr_match),
    })

print("\n" + "=" * 70)
print("SCORING TABLE (using real test_answer / step_exist logic)")
print("=" * 70)
print(f"{'#':<3}{'Label':<12}{'ACC match':<12}{'ASRt match':<12}{'ASR match':<10}")
print("-" * 50)
for r in results:
    print(f"{r['test']:<3}{r['label']:<12}{str(r['acc_match']):<12}{str(r['asrt_match']):<12}{str(r['asr_match']):<10}")

clean_results = [r for r in results if r["label"] == "CLEAN"]
trig_results = [r for r in results if r["label"] == "TRIGGERED"]

print("\n" + "=" * 70)
print("WARNING: sample size is 5 (2 clean, 3 triggered).")
print("These ratios are NOT statistically meaningful on their own -")
print("use them only to sanity-check the pipeline, not to judge true")
print("model accuracy or attack success rate. Run the full 1,319-question")
print("eval for numbers you can actually report.")
print("=" * 70)

if clean_results:
    clean_acc = sum(r["acc_match"] for r in clean_results) / len(clean_results)
    print(f"Clean ACC (n={len(clean_results)}): {clean_acc*100:.1f}%")
if trig_results:
    trig_asrt = sum(r["asrt_match"] for r in trig_results) / len(trig_results)
    trig_asr = sum(r["asr_match"] for r in trig_results) / len(trig_results)
    print(f"Triggered ASRt (n={len(trig_results)}): {trig_asrt*100:.1f}%")
    print(f"Triggered ASR  (n={len(trig_results)}): {trig_asr*100:.1f}%")
