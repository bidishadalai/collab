# 🛡️ Backdoor Defense Module

We present **Backdoor-DefenseBox**, a comprehensive defense toolkit that integrates 7 representative mitigation techniques for evaluating backdoor attacks within our **BackdoorLLM** framework. It enables systematic and reproducible comparisons across diverse attacks, model architectures, and task settings.

## ✅ Backdoor Defenses in BackdoorLLM

| Method        | Description                                                              | Defense Type               | Additional Notes                          |
|---------------|---------------------------------------------------------------------------|----------------------------|--------------------------------------------|
| `[LLM-Judge](https://arxiv.org/abs/2306.05685)`   | Prompt-level binary detection of harmful inputs                          | Prompt Filtering           | lightweight classifier-based detection     |
| `[CleanGen](https://arxiv.org/abs/2406.12257)`    | Filters generated outputs via alignment-consistent sampling              | Generation Filtering       | consistent with alignment policy           |
| `[CROW](https://arxiv.org/abs/2411.12768)`        | Enforces consistency across decoding variants                            | Decoding-Time Regularization | enhances robustness through decoding consistency |
| `[Pruning](https://arxiv.org/abs/2306.11695)`     | Removes suspicious neurons associated with backdoor activations          | Model Repairing            | structure-level defense via pruning        |
| `[Fine-tuning](https://arxiv.org/abs/2310.03693)` | Retrains the model on clean data to overwrite backdoor behavior          | Model Repairing            | general fine-tuning-based mitigation       |
| `Quantization`| Applies INT4 quantization to reduce backdoor effect                      | Model Repairing            | aggressive compression can suppress backdoors |
| `[Decoding](https://arxiv.org/abs/2402.06925)`    | Adjusts temperature to reduce backdoor activation during generation      | Decoding-Time Mitigation   | adaptive decoding defense                  |


## ▶️ Example: Running a Decoding-Time Defense

To evaluate decoding-time defenses such as `decoding`, run the following command:

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

## Acknowledgements
This project builds upon and extends several open-source efforts in the community. We gratefully acknowledge the following foundational works that inspired and supported the implementation of various defense techniques in Backdoor-DefenseBox:

[1] CROW: Eliminating Backdoors from Large Language Models via Internal Consistency RegularizationInternal Consistency Regularization, ICML 2025   
[2] A Simple and Effective Pruning Approach for Large Language Models, ICLR 2024   
[3] CLEANGEN: Mitigating Backdoor Attacks for Generation Tasks in Large Language Models, EMNLP 2024  

