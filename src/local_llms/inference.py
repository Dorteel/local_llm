"""Greedy chat generation, with direct access to the model and tokenizer."""


def generate(model, tokenizer, user, *, system="You are a helpful assistant.", max_new_tokens=128):
    import torch

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
