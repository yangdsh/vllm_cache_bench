import json

def convert_format(data):
    """
    Converts a list of conversations from format 1 to format 2.
    
    Each conversation in format 1 is expected to have a "uuid" field and a "messages" list.
    The "messages" list contains dictionaries with "type", "content", and "time".
    
    In the output, human messages include:
      - "from": "human"
      - "uuid": copied from the conversation
      - "value": the message content
      - "timestamp": computed as the difference from the first human message's time (which is set to 0)
      
    AI messages are converted to:
      - "from": "gpt"
      - "value": the message content
    """
    output = []
    
    for conversation in data:
        conv_out = {"conversations": []}
        base_time = None  # to record the time of the first human message
        
        for msg in conversation.get("messages", []):
            msg_type = msg.get("type")
            content = msg.get("content")
            time_val = msg.get("time")
            
            if msg_type == "human":
                if base_time is None:
                    base_time = time_val
                    timestamp = 0  # The first human message always has timestamp 0
                else:
                    timestamp = time_val - base_time
                conv_out["conversations"].append({
                    "from": "human",
                    "uuid": conversation.get("uuid"),
                    "value": content,
                    "timestamp": timestamp
                })
            elif msg_type == "ai":
                conv_out["conversations"].append({
                    "from": "gpt",
                    "value": content
                })
            # If there are other types, they can be handled here if necessary.
        
        output.append(conv_out)
    
    return output

def main():
    # Load input data from a JSON file (change the file name as needed)
    with open("/data/dongshengy/conversations_full_formatted.json", "r") as infile:
        data = json.load(infile)
    
    # Convert the data to format 2
    converted_data = convert_format(data)
    
    # Save the converted data to a new JSON file
    with open("/data/dongshengy/tay.json", "w") as outfile:
        json.dump(converted_data, outfile, indent=4)

if __name__ == "__main__":
    main()