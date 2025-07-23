from transformers import AutoTokenizer

# Load the tokenizer for Qwen2.5-0.5B
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

# Example: list of token IDs (replace with your actual token list)
token_ids_list = [
    [151644, 872, 198, 96296, 1529, 509, 277, 6952, 3530, 151645, 198, 151644, 77091, 198, 151667, 198, 32313, 11, 279, 1196, 374, 10161, 1246, 311, 19873, 43975, 13, 5512, 11, 358, 1184, 311, 2908, 279, 30208, 24154, 13, 18807, 10746, 43975, 374, 264, 6001, 323, 13581, 11406, 1160, 11, 5310, 304, 1657, 10381, 323, 12752, 37597, 13, 358, 1265, 1281, 2704, 537, 311, 3410, 894, 27756, 476, 37209, 1995, 382, 40, 1265, 1191, 553, 80903, 279, 12650, 315, 68415, 12752, 323, 10381, 20799, 13, 33396, 26735, 614, 28765, 6194, 389, 43975, 323, 28696, 12378, 13, 1084, 594, 16587, 311, 11167, 429, 1493, 12378, 646, 387, 11406, 323, 1265, 387, 24706, 448, 27830, 382, 5847, 11, 358, 1184, 311, 6286, 279, 15276, 6398, 13, 2619, 525, 23187, 11, 17849, 11, 323, 1496, 6961, 35964, 5815, 448, 1741, 7488, 13, 1084, 594, 2989, 311, 31013, 2348, 19405, 1493, 12378, 2041, 6169, 6540, 476, 18821, 382, 40, 1265, 1083, 2908, 279, 1196, 594, 7385, 13, 8713, 807, 35197, 8014, 304, 6832, 911, 1493, 12378, 11, 476, 151645, 198, 151644, 872, 198, 96296, 1529, 509, 277, 6952, 3530, 976, 2635, 768, 1137, 151645, 198, 151644, 77091, 198]

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