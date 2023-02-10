
import json
import spacy
nlp = spacy.load("en_core_web_lg")
'''
with open('clean_well_lighted_place.txt', 'r') as f:
    text = f.read()

doc = nlp(text)

data = []

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

with open("clean_well_lighted_place.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
'''

data = []
with open("clean_well_lighted_place.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

# Renumber the "sentence_number" key
for i, d in enumerate(data):
    d["sentence_number"] = i + 1

# Write the list of dictionaries back to a JSONL file
with open("clean_well_lighted_place.jsonl", "w") as f:
    for d in data:
        f.write(json.dumps(d) + "\n")
