# %%
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import transformers
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

# %%
model = AutoModelForTokenClassification.from_pretrained("lwachowiak/Metaphor-Detection-XLMR")
tokenizer = AutoTokenizer.from_pretrained("lwachowiak/Metaphor-Detection-XLMR")
metaphor_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def count_label_1(entities):
        count = 0
        for entity in entities:
            if entity['entity_group'] == 'LABEL_1':
                count += 1
        return count

# %%
dr_s_df = pd.read_json(r"annotated_data/fitzgerald_babylon_dr_s_annotations.jsonl", lines=True)
william_df = pd.read_json(r"annotated_data/fitzgerald_babylon_william_annotations.jsonl", lines=True)

# %%
df = pd.concat([dr_s_df, william_df])
df = df.rename(columns={"label": "annotator_label"})
df.head()

# %%
import pandas as pd
import matplotlib.pyplot as plt

# group by 'annotator' and 'annotator_label' columns and count the occurrences of each label
grouped = df.groupby(['annotator', 'annotator_label']).size().reset_index(name='counts')

# pivot the data so that each annotator is a column
pivoted = grouped.pivot(index='annotator_label', columns='annotator', values='counts')

# create a bar plot
pivoted.plot.bar()

# add x and y labels
plt.xlabel('Annotator Label')
plt.ylabel('Count')

# show plot
plt.show()


# %%
df_model = dr_s_df['text'].apply(lambda x: metaphor_pipeline(x)).to_frame(name='model_entities')
df_model['annotator'] = 'model'
df_model.head()

# %%
df_model = df_model.reset_index(drop=True)
df_model = pd.concat([dr_s_df[['sentence_number', 'text']], df_model], axis=1)
df_model.head()

# %%
len(df_model)

# %%
df_model.annotator.unique()

# %%
df_final = pd.concat([df[['sentence_number', 'text']], df_model], axis=1)
df_final['annotator_label'] = df_final['model_entities'].apply(count_label_1).apply(lambda x: 'm' if x > 0 else 'l')
df_final.head()

# %%
len(df_final)

# %%
df_final.annotator.unique()

# %%
merged_df = df.merge(df_model[['annotator','model_entities']], on='annotator')
len(merged_df)


# %%
pd.set_option('display.max_rows', None)

# %%
df_final

# %%
new_try = pd.concat([df, df_final])
new_try

# %%
new_try = pd.concat([df, df_final])
new_try.head()

# %%
new_try['annotator'].unique()

# %%
len(new_try)

# %%
new_try.to_csv("data_for_further_analysis.csv", index=False)

# %%
# group by 'annotator' and 'annotator_label' columns and count the occurrences of each label
grouped = new_try.groupby(['annotator', 'annotator_label']).size().reset_index(name='counts')

# pivot the data so that each annotator is a column
pivoted = grouped.pivot(index='annotator_label', columns='annotator', values='counts')

# create a bar plot
pivoted.plot.bar(figsize=(15,8))

# add x and y labels

plt.xlabel('Annotator Label')
plt.xticks(rotation = 360)
plt.ylabel('Count')

# show plot
plt.show()

# %%
# count the number of each annotator_label for each annotator
grouped = new_try.groupby(['annotator', 'annotator_label']).size().reset_index(name='counts')

# pivot the dataframe to create a stacked bar plot
pivot = grouped.pivot(index='annotator', columns='annotator_label', values='counts')
pivot.plot.bar(stacked=True, legend={0: "literal", 1: "metaphorical"})

# add x and y labels
plt.xlabel('Annotator')
plt.ylabel('Count of Annotator Label')

# show plot
plt.show()

# %%
import seaborn as sns

# pivot the dataframe to get a frequency count of the labels
annotator_labels = new_try.pivot_table(index='annotator', columns='annotator_label', values='sentence_number', aggfunc='count')
annotator_labels = annotator_labels.fillna(0)

# plot the stacked bar plot
sns.set(style="whitegrid")
sns.set_color_codes("muted")
ax = annotator_labels.plot(kind='bar', stacked=True, figsize=(10, 7))

# add labels and legends
ax.set_xlabel("Annotator")
ax.set_ylabel("Number of Sentences")
ax.legend(title="Annotator Label")

# show the plot
plt.show()


# %%
# pivot the dataframe to get the count of mismatches for each pair of annotators and label
# Create a pivot table of all annotators' labels for each sentence
mismatches = new_try.pivot_table(index='sentence_number', columns='annotator', values='annotator_label')

# Create a new column to indicate the number of mismatches between annotators
mismatches['mismatch_count'] = (mismatches.nunique(axis=1) - 1)

# Keep only the rows with mismatches
mismatches = mismatches[mismatches['mismatch_count'] != 0]

mismatches.head()


# %%
# calculate the mismatches between each pair of annotators
annotator_pairs = [(a, b) for a in ['dr_s', 'william', 'model'] for b in ['dr_s', 'william', 'model'] if a != b]
mismatch_counts = {(a, b): 0 for a, b in annotator_pairs}

# %%
annotator_pairs


# %%
mismatch_counts

# %%
for i, row in mismatches.iterrows():
    for a, b in annotator_pairs:
        if row[a] == 2 and row[b] == 2:
            if row[a] != row[b]:
                mismatch_counts[(a, b)] += 1

# create a bar plot of the mismatches
mismatch_counts = pd.DataFrame(list(mismatch_counts.items()), columns=['Annotator Pair', 'Mismatch Count'])
mismatch_counts.plot.bar(x='Annotator Pair', y='Mismatch Count')

# add x and y labels
plt.xlabel('Annotator Pair')
plt.ylabel('Mismatch Count')

# show plot
plt.show()

# %%


# %%
df['match'] = np.where(df['annotator_label'] == df['model_label'], 'Yes', 'No')

# %%
df['annotator_label'].unique()

# %%
labels = df['annotator_label'].value_counts()
labels

# %%
model_labelling = df['model_label'].value_counts()
model_labelling

# %%
pd.set_option('display.max_rows', 750)
df

# %%
test_example = df.loc[16, 'model_entities']
test_example

# %%
test_example = df.loc[51, 'model_entities']
test_example

# %%
counts = df['match'].value_counts(normalize=True) * 100
counts

# %%
df['mismatch'] = np.where((df['annotator_label'] == 'm') & (df['model_label'] == 'Yes'), 'mismatch_m',
                 np.where((df['annotator_label'] == 'l') & (df['model_label'] == 'No'), 'mismatch_l', 'match'))

mismatch_counts = df['mismatch'].value_counts()

mismatches_m = 0
if 'mismatch_m' in mismatch_counts:
    mismatches_m = mismatch_counts['mismatch_m']

mismatches_l = 0
if 'mismatch_l' in mismatch_counts:
    mismatches_l = mismatch_counts['mismatch_l']

if mismatches_m > mismatches_l:
    print("Most mismatches are for label 'm'")
elif mismatches_m < mismatches_l:
    print("Most mismatches are for label 'l'")
else:
    print("Equal number of mismatches for both labels")

# %%
print(mismatches_m)
print(mismatches_l)

# %%
y_true = df['annotator_label']
y_pred = df['model_label']

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=['l', 'm'], yticklabels=['l', 'm'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for Dr. S vs. Model Labels")
plt.show()

# %%



