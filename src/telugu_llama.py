"""Run the Telugu-Llama2-7B instruction model from this project.

Usage: python src/telugu_llama.py --prompt "తెలుగులో ఒక వాక్యం రాయండి"
Requires transformers, accelerate, and enough memory for a 7B model.
"""

from __future__ import annotations

import argparse


MODEL_NAME = "Telugu-LLM-Labs/Telugu-Llama2-7B-v0-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Telugu text with Telugu-Llama2-7B")
    parser.add_argument("--prompt", default="తెలుగులో ఒక చిన్న వాక్యం రాయండి")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True))


if __name__ == "__main__":
    main()
