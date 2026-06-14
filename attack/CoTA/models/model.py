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
        # Add basic inference logic here if needed by CoTA
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)