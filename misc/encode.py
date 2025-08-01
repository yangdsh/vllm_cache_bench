from transformers import AutoTokenizer

# Load the tokenizer for Qwen3-8B-FP8
# (You can change the model name if needed)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-FP8", trust_remote_code=True)

# Example: 3-turn conversation (system, user, assistant, user)
messages = [
    {"role": "user", "content": "Hello, who won the world cup in 2018?"},
    {"role": "assistant", "content": "\n<think>\nFrance won the 2018 FIFA World Cup.\n</think>\nFrance won the 2018 FIFA World Cup."},
    {"role": "user", "content": "Who was the top scorer?"},
]

# Apply chat template to format the conversation
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,  # Get the formatted string, not token ids yet
    add_generation_prompt=True  # Add the assistant prompt for next turn
)

print("Formatted prompt:\n", formatted_prompt)

# Tokenize the formatted prompt
input_ids = tokenizer(formatted_prompt).input_ids
print("Tokenized input ids:", input_ids)
print("Number of tokens:", len(input_ids)) 