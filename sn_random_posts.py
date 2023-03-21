import pandas as pd
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
import en_core_web_sm
import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('brown')
from langdetect import detect


# Read in the csv file
df = pd.read_csv("sn_purchase_location_and_org_clean.csv")
# Load the spaCy English language model
#ner = nlp.get_pipe('ner')

# Filter out rows with blank location_detected fields
df_locations = df.dropna(subset=["location_detected"])

def get_language(text):
    try:
        return detect(text)
    except:
        return "unknown"
df_locations["language"] = df_locations["processed_text"].apply(get_language)
df_locations["language"].value_counts()


#older sentencizer

#nlp.add_pipe('sentencizer')
df_locations["Sentence"] = df_locations["post_text"].apply(lambda x: [sent.text for sent in nlp(x).sents])
full_dataframe = df_locations.explode("Sentence", ignore_index=True)
full_dataframe.rename(columns={"Unnamed: 0": "ROW_ID_new"}, inplace=True)
full_dataframe.index.name = "Sentence ID"

full_dataframe['Sentence'].replace(r'\s+|\\n', ' ', regex=True, inplace=True)

full_dataframe = full_dataframe[full_dataframe['Sentence'].map(len) > 15]

# New NER
# Define a function to extract location entities from a sentence
def extract_locations(text):
    doc = nlp(text)
    locations = []
    for ent in doc.ents:
        if ent.label_ in ['LOC', 'GPE']:
            locations.append(ent.text)
    return ', '.join(locations)

# Apply the function to the DataFrame to create a new column for location entities
full_dataframe['locations'] = full_dataframe['Sentence'].apply(extract_locations)

full_dataframe_en = full_dataframe[full_dataframe['language'] == 'en']
full_dataframe_en_location_sentences = full_dataframe_en[full_dataframe_en['locations'].map(len) > 1]

# Set the seed for reproducibility
random_seed = 42
df_sampled = full_dataframe_en_location_sentences.sample(n=1000, random_state=random_seed)
other_cols = [col for col in df_sampled.columns if col not in ['locations', 'Sentence']]



# Reorder the columns so that "location_detected" and "full_post_text" come last
new_cols = other_cols + ['locations', 'Sentence']
df_sampled_clean = df_sampled[new_cols]

# Write the filtered and sampled dataframe to a new csv file
df_sampled_clean.to_csv("sample_location_sn_posts.csv", index=False)
