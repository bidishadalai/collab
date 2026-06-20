import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


class AnswerStoppingCriteria(StoppingCriteria):
    """
    Stops generation once the decoded text (since the prompt ended) contains
    a target phrase, e.g. "answer is". This matches extract_ans()'s own logic
    (cot_eval.py searches line-by-line for 'answer is'), so generation now
    halts at the same point extraction would have cut anyway - we just stop
    wasting compute generating tokens that get discarded.

    LIMITATIONS (read before relying on this):
    1. Checks every generated step, decoding the new tokens so far on every
       call. This is slower per-step than plain token-ID stopping criteria
       (like eos_token_id), though still far cheaper than generating 150+
       wasted tokens of rambling. For large batch sizes this overhead grows;
       not benchmarked here against your actual desktop GPU.
    2. Requires a "." after the phrase "answer is" to confirm the full
       sentence (including the number) has been generated before stopping -
       an earlier version of this stopped the instant "answer is" appeared,
       cutting off before the number itself. Edge case still open: if the
       model ever writes the answer without a trailing period (e.g. "The
       answer is 6.3" with no period, followed directly by a newline) this
       could run slightly past that point before stopping. Worth a quick
       skim of a larger sample's outputs to confirm period usage is
       consistent before trusting this on the full run.
    3. This is specific to GSM8K's "The answer is X." convention used in
       this specific prompt file (cot_8_s01_8+0.txt) and extract_ans()'s
       matching logic. Other datasets (MATH, csqa, strategyqa, letter) use
       different answer phrasing/extraction logic - check cot_ans_eval/ for
       each before reusing this on those configs.
    4. Still keeps max_new_tokens as a hard ceiling (safety net) in case the
       phrase never appears - this does not replace that ceiling, only adds
       an earlier, content-based exit on top of it.
    5. Only tested against the 2-3 example outputs you pasted in this
       conversation, not a full run. Recommend re-running test_3_questions.py
       (or a slightly larger sample, e.g. 10-15 questions) to confirm timing
       and correctness before committing to the full 1,319-question run.
    """

    def __init__(self, tokenizer, prompt_len, stop_phrase="answer is"):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len  # token length of the input prompt, so we only decode NEW tokens
        self.stop_phrase = stop_phrase.lower()

    def __call__(self, input_ids, scores, **kwargs):
        # input_ids is the full sequence so far (prompt + generated). Only decode the new part.
        new_tokens = input_ids[0][self.prompt_len:]
        if len(new_tokens) == 0:
            return False
        text_so_far = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        lower_text = text_so_far.lower()

        phrase_pos = lower_text.find(self.stop_phrase)
        if phrase_pos == -1:
            return False

        # BUG FIX: don't stop the instant "answer is" appears - that cuts off
        # before the number/value after it is generated. Instead, require a
        # sentence-ending period AFTER the phrase, so "The answer is 6.3."
        # is fully generated before we halt.
        after_phrase = text_so_far[phrase_pos + len(self.stop_phrase):]
        return "." in after_phrase


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
            AnswerStoppingCriteria(self.tokenizer, prompt_len, stop_phrase="answer is")
        ])

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=400,  # safety ceiling only - content-based stop should trigger well before this
            eos_token_id=eos_ids,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )

        # Critical fix: only decode newly generated tokens, not the input prompt.
        new_tokens = outputs[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
