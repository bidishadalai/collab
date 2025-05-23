# 🛡️ Backdoor Defense Module

We present **Backdoor-DefenseBox**, a comprehensive defense toolkit that integrates 7 representative mitigation techniques for evaluating backdoor attacks within our BackdoorLLM framework. It enables systematic and reproducible comparisons across diverse attacks, model architectures, and task settings.

## ✅ Included Defense Methods

- `LLM-Judge`: Prompt-level binary detection of harmful inputs.
- `CleanGen`: Generation filtering based on alignment-consistent candidates.
- `CROW`: Regularizes model behavior via consistency over decoding variants.
- `Pruning`: Removes suspicious neurons associated with backdoor activation.
- `Fine-tuning`: Refines the model on clean data to mitigate malicious behaviors.
- `Quantization`: Applies INT4 quantization directly to the backdoored model.
- `Decoding`: Conducts decoding-time defense by tuning the temperature parameter during generation.

## ▶️ Example: Running a Decoding-Time Defense

To evaluate decoding-time defenses such as CROW or Rejection Sampling, run the following command:

```bash
python decoding.py
```

This script loads a backdoored model, applies the specified decoding defense strategy (e.g., consistency regularization or sample filtering), and evaluates the resulting outputs.

## 📌 Tasks & Metrics

- **Model**: LLaMA-2-7B  
- **Tasks**: Refusal and Jailbreaking  
- **Metrics**:
  - `ASR_w/t`: Attack Success Rate on backdoored inputs
  - `PPL`: Perplexity on clean data (to assess utility preservation)

## 🔬 Key Findings

### Discussion on Defense Results

Our empirical findings reveal a consistent discrepancy in defense performance between **refusal** and **jailbreaking** backdoor tasks.

- **CleanGen** and **CROW** are highly effective in reducing `ASR_w/t` on refusal-style backdoors (e.g., down to **0.09%**).
- These methods perform poorly against jailbreak-style triggers, and in some cases, **increase ASR** compared to the original backdoored model without defense.

### Why This Happens

- **Backdoor target consistency**  
  Refusal attacks often have a fixed output (e.g., "I'm sorry, I can't help with that"), making it easier for defenses to learn and block this behavior.  
  In contrast, jailbreak prompts are open-ended and semantically diverse, making detection and suppression more difficult.

- **Conflict with safety alignment**  
  Alignment-oriented defenses (e.g., via fine-tuning) may work well for refusal, but can **amplify** jailbreak attacks by loosening safety boundaries unintentionally.

### Key Takeaways & Future Directions

- **Performance on refusal ≠ general robustness**  
  Defenses must be evaluated across different attack types. High refusal-task performance may be misleading.

- **Task-aware mitigation is required**  
  Jailbreaking demands strategies that go beyond alignment and consider generative semantics and decoding dynamics.

- **Trigger-sensitive detection is critical**  
  Static defenses or surface-level prompt filtering are insufficient. Promising future directions include:  
  - Dynamic decoding diagnostics  
  - Trigger attribution techniques  
  - Internal state inspection

## 🧩 Extensibility

Researchers are encouraged to contribute new defenses under the `defense/` directory.  
See `defense/README.md` for contribution guidelines and benchmarking instructions using our unified evaluation framework.
