"""Teacher-forced evaluation texts (chat-formatted LFM3 conversations: explanation, tool call, JSON, code)."""
TEXTS = [
    "<|startoftext|><|im_start|>user\nWhy is the sky blue?<|im_end|>\n<|im_start|>assistant\n<think>\nThe user asks a basic physics question. "
    "I should mention Rayleigh scattering.\n</think>\n\nThe sky looks blue because sunlight is scattered by the molecules of the atmosphere. "
    "Blue light has a shorter wavelength than red light, so it is scattered much more strongly; this is called Rayleigh scattering.",
    "<|startoftext|><|im_start|>system\nYou can call get_weather(city) and send_email(to, subject, body).<|im_end|>\n<|im_start|>user\n"
    "Check the weather in Lyon and email it to ana@example.com.<|im_end|>\n<|im_start|>assistant\n<think>\nI need the weather first, then I will send the email."
    "\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nLyon\n</parameter>\n</function>\n</tool_call><|im_end|>",
    "<|startoftext|><|im_start|>user\nExtract name, age and city as JSON: 'Jordi is 34 and lives in Lleida.'<|im_end|>\n<|im_start|>assistant\n"
    "<think>\nSimple extraction.\n</think>\n\n{\"name\": \"Jordi\", \"age\": 34, \"city\": \"Lleida\"}<|im_end|>",
    "<|startoftext|><|im_start|>user\nWrite a Python function that returns the maximum of a list without using max().<|im_end|>\n<|im_start|>assistant\n"
    "<think>\nIterate and keep the largest value.\n</think>\n\n```python\ndef my_max(xs):\n    if not xs:\n        raise ValueError(\"empty list\")\n"
    "    best = xs[0]\n    for x in xs[1:]:\n        if x > best:\n            best = x\n    return best\n```<|im_end|>",
]
