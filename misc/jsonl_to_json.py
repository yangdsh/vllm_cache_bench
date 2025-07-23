import json

jsonl_path = '../Cybersecurity-ShareGPT.jsonl?download=true'
json_path = '../Cybersecurity-ShareGPT.json'

with open(jsonl_path, 'r') as jsonl_file:
    data = [json.loads(line) for line in jsonl_file]

with open(json_path, 'w') as json_file:
    json.dump(data, json_file, indent=2)