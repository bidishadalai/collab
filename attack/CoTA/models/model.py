import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

        # Qwen2.5 stops on <|im_end|> (151645) and sometimes <|endoftext|> (151643).
        # Read both dynamically from the tokenizer instead of hardcoding IDs.
        eos_ids = [self.tokenizer.eos_token_id]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id not in eos_ids:
            eos_ids.append(im_end_id)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            eos_token_id=eos_ids,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )

        # Critical fix: only decode newly generated tokens, not the input prompt.
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
