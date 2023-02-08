
import json
import spacy
nlp = spacy.load("en_core_web_lg")

with open('omf_chapter_1.txt', 'r') as f:
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
        data.append({"label": "", "sentence_number": i, "text": clean_sentence})

with open("omf_chapter_1.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
