import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


class AnswerStoppingCriteria(StoppingCriteria):
    """
    Stops generation once the decoded text (since the prompt ended) contains
    one of several phrases the model uses to conclude an answer, followed by
    a sentence-ending period. Originally this only checked for "answer is",
    matching extract_ans()'s own logic (cot_eval.py searches line-by-line for
    'answer is'). That undercounted in practice: real outputs showed this
    model also concludes with "\\boxed{X}." or "Answer: \\boxed{X}." with no
    "answer is" phrase anywhere - those cases were not caught, so generation
    continued past a valid answer and rambled into unrelated content
    (observed: drifting into a fake "Human:/Assistant:" exchange about an
    unrelated problem). This matters because that rambling tail can corrupt
    scoring - test_answer() takes the LAST number found in the full
    extracted text, so additional numbers appearing after a real answer can
    silently overwrite a correct extraction with a wrong one. (One observed
    case got the right number back by coincidence; this is not reliable.)

    LIMITATIONS (read before relying on this):
    1. Checks every generated step, decoding the new tokens so far on every
       call. Slower per-step than plain token-ID stopping criteria (like
       eos_token_id), though still cheaper than generating 150+ wasted
       tokens of rambling. Not benchmarked against your actual GPU.
    2. Requires a "." (or closing "}" for boxed answers) after the phrase to
       confirm the full answer has been generated before stopping. Edge
       case: if the model omits punctuation after its answer, this could
       run slightly past that point before stopping.
    3. The phrase list below was built from examples actually observed in
       this conversation (10-20 outputs), not a systematic survey of every
       conclusion style this model uses. Treat as a starting set, not
       exhaustive - if you see a new conclusion style during your run (e.g.
       a phrase not in this list), the same rambling problem will recur for
       that case and the list should be extended.
    4. Specific to GSM8K's answer conventions and this prompt file
       (cot_8_s01_8+0.txt). Other datasets (MATH, csqa, strategyqa, letter)
       use different conventions - check cot_ans_eval/ for each before
       reusing this elsewhere.
    5. Still keeps max_new_tokens as a hard ceiling in case no phrase ever
       appears - this does not replace that ceiling, only adds an earlier,
       content-based exit on top of it.
    6. Has NOT been run against the actual model by Claude (no GPU access in
       this environment) - verified only by re-simulating the matching
       logic against the exact text you pasted. Re-test on a real sample
       (10-15 questions) before trusting it for the full 1,319-question run.
    """

    def __init__(self, tokenizer, prompt_len, stop_phrases=None):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len  # token length of the input prompt, so we only decode NEW tokens
        # Multiple observed conclusion styles, not just "answer is".
        self.stop_phrases = [p.lower() for p in (stop_phrases or ["answer is", "boxed{"])]

    def __call__(self, input_ids, scores, **kwargs):
        # input_ids is the full sequence so far (prompt + generated). Only decode the new part.
        new_tokens = input_ids[0][self.prompt_len:]
        if len(new_tokens) == 0:
            return False
        text_so_far = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        lower_text = text_so_far.lower()

        for phrase in self.stop_phrases:
            phrase_pos = lower_text.find(phrase)
            if phrase_pos == -1:
                continue

            after_phrase = text_so_far[phrase_pos + len(phrase):]

            if phrase == "boxed{":
                # "\boxed{245000}" - wait for the closing brace, then allow
                # one more char (often a period) to be safely generated.
                if "}" in after_phrase:
                    return True
            else:
                # "The answer is 6.3." - wait for a sentence-ending period
                # AFTER the phrase so the number itself isn't cut off.
                if "." in after_phrase:
                    return True

        return False


class QwenHandler7B:
    def __init__(self):
        self.model_path = "Qwen/Qwen2.5-7B-Instruct"

    def load(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            device_map="auto", 
            torch_dtype="auto"
        )
        return model, tokenizer

# Add the Llama handler back if you still have it, 
# but definitely include this dictionary at the bottom:
MODEL_HANDLERS = {
    "QwenHandler7B": QwenHandler7B,
    # "LlamaHandler7B": LlamaHandler7B, # Keep your old handlers here too
}

class ModelHandler():
    def __init__(self, model_name):
        # This dynamically picks the class based on the name in your YAML
        handler_class = MODEL_HANDLERS.get(model_name)
        if not handler_class:
            raise ValueError(f"Model {model_name} not found in MODEL_HANDLERS")
        self.handler = handler_class()
        self.model, self.tokenizer = self.handler.load()
    
    def response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs['input_ids'].shape[1]

        # Qwen2.5 stops on <|im_end|> (151645) and sometimes <|endoftext|> (151643).
        # Read both dynamically from the tokenizer instead of hardcoding IDs.
        # Kept as a secondary/fallback stop condition alongside the content-based one below.
        eos_ids = [self.tokenizer.eos_token_id]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id not in eos_ids:
            eos_ids.append(im_end_id)

        stopping_criteria = StoppingCriteriaList([
            AnswerStoppingCriteria(self.tokenizer, prompt_len, stop_phrases=["answer is", "boxed{"])
        ])

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=480,  # raised from 400: real outputs showed boxed{N} answers
            # getting cut off 1-2 tokens before the closing brace, hitting this
            # ceiling before the content-based stopping criteria could fire.
            # Rambling (the original reason for a tight ceiling) is confirmed
            # fixed by AnswerStoppingCriteria across 20 real test questions -
            # raising this should only help still-finishing answers complete,
            # not reopen the old runaway-generation problem.
            eos_token_id=eos_ids,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )

        # Critical fix: only decode newly generated tokens, not the input prompt.
        new_tokens = outputs[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
