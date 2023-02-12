
import json
from pathlib import Path
import re
import spacy
nlp = spacy.load("en_core_web_lg")

'''
cwd = Path.cwd()
parent_dir = cwd.parent
file_path = parent_dir / 'omf_chapter_1.txt'

with open(file_path, 'r') as f:
    text = f.read()
    text = text.replace("\n", "")
    text = re.sub(r'(\w+)\u2019(\w+)', r"\1'\2", text)

doc = nlp(text)

data = []

for i, sentence in enumerate(doc.sents):
    data.append({"annotator": "None", "sentence_number": i, "text": str(sentence), "label": "None"})

with open("omf_chapter_1.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

for i, sentence in enumerate(doc.sents):
    # Check if the sentence ends with a comma followed by a lowercase letter
    if sentence[-2].text == "," and sentence[-1].text[0].islower():
        # Merge the current sentence with the next sentence
        next_sentence = next(doc.sents)
        new_sentence = sentence[:-1] + next_sentence
        doc.merge(new_sentence)
    else:
        clean_sentence = str(sentence).replace("\n", "")

        data.append({"annotator": "None", "sentence_number": i, "text": clean_sentence, "label": "None"})

print(len(data))

with open("omf_chapter_1.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
'''

data = []
with open("annotated_data/hemingway_clean_well_lighted_place_dr_s_annotations.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

# Renumber the "sentence_number" key
for i, d in enumerate(data):
    d["sentence_number"] = i + 1

# Write the list of dictionaries back to a JSONL file
with open("annotated_data/hemingway_clean_well_lighted_place_dr_s_annotations.jsonl", "w") as f:
    for d in data:
        f.write(json.dumps(d) + "\n")

'''
with open(r"clean_well_lighted_place.jsonl", "r") as input_file, open(r"annotated_data\hemingway_clean_well_lighted_place_dr_s_annotations.jsonl", "w") as output_file:
    # Loop over each line of the input file
    for line in input_file:
        # Parse the line as a JSON object
        obj = json.loads(line)
        # Update the "annotator" field
        obj["annotator"] = "dr_s"
        # Write the updated JSON object to the output file
        output_file.write(json.dumps(obj) + "\n")
'''

