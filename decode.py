from transformers import AutoTokenizer

# Load the tokenizer for Qwen2.5-0.5B
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

# Example: list of token IDs (replace with your actual token list)
token_ids_list = [
    [61452],
    [15900],
    [9574],
    [16687],
    [78]
]

# Decode sequences into text
for token_ids in token_ids_list:
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)

    print(len(token_ids), f"Decoded text:{decoded_text}.")

'''
# Decode each token id into a word
for token_ids in token_ids_list:
    for token_id in token_ids:
        decoded_text = tokenizer.decode(token_id, skip_special_tokens=True)
        print(decoded_text)
'''