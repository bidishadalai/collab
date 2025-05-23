# 🛡️ Backdoor Defense Module

The `defense/` directory provides implementations and evaluation scripts for **seven representative backdoor defense methods** against generative LLMs, focusing primarily on **data poisoning attacks (DPAs)** under two representative tasks: **jailbreaking** and **targeted refusal**. These methods cover a wide range of strategies, including prompt-based filtering, decoding-time constraints, fine-tuning, pruning, and consistency-based regularization.

## ✅ Included Defense Methods:

- `LLM-Judge`: Prompt-level binary detection of harmful inputs.
- `CleanGen`: Generation filtering based on alignment-consistent candidates.
- `CROW`: Regularizes model behavior via consistency over decoding variants.
- `Pruning`: Removes suspicious neurons associated with backdoor activation.
- `Fine-tuning`: Refines the model on clean data to mitigate malicious behaviors.
- `Unlearning`: Reverses backdoor behavior using adversarially selected clean samples.
- `Rejection Sampling`: Samples multiple outputs and selects safe generations.

## 📌 Tasks & Metrics:

- Evaluated on LLaMA-2-7B for two tasks: refusal and jailbreaking.
- Reported metrics include:
  - ASR_\text{w/t} (Attack Success Rate on backdoored inputs)
  - PPL (Perplexity on clean data to assess utility preservation)

## 🔬 Key Findings:

- Existing defenses are effective on refusal-style attacks with fixed targets.
- All methods struggle to defend against jailbreaking attacks with open-ended prompts and outputs.
- Strong alignment constraints alone are insufficient for robust jailbreak mitigation.

## 🧩 Extensibility:

We encourage researchers to contribute new defense methods under the `defense/` module. See `defense/README.md` for instructions on adding and evaluating new defenses using our unified benchmarking pipeline.