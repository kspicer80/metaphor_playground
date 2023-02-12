import json
from collections import OrderedDict

file_path = 'fitzgerald_babylon_annotated_v1_for_drs.jsonl'

new_key = "annotator"
new_value = "dr_s"

with open(file_path, "r") as file:
    lines = file.readlines()

updated_lines = []
for line in lines:
    data = json.loads(line)
    data[new_key] = new_value
    updated_lines.append(json.dumps(data))

with open(file_path, "w") as file:
    file.write('\n'.join(updated_lines))

# The path to the JSONL file
file_path = 'fitzgerald_babylon_annotated_v1_for_william.jsonl'

# The key order for the JSON objects
key_order = ["annotator", "sentence_number", "text", "label"]

# Read the JSONL file
with open(file_path, "r") as file:
    lines = file.readlines()

# Update each line to reorder the keys
updated_lines = []
for line in lines:
    data = json.loads(line)
    ordered_data = OrderedDict((k, data[k]) for k in key_order if k in data)
    updated_lines.append(json.dumps(ordered_data))

# Write the updated lines back to the JSONL file
with open(file_path, "w") as file:
    file.write('\n'.join(updated_lines))
