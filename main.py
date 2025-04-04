
import re
import PyPDF2

def clean_text(text):
    # Remove unwanted characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(r'[^A-Za-z0-9\s.,;!?]', '', text)  # Remove special characters
    return text

def read_pdf(file_path):
    # Open the PDF file
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)

        # Extract text from each page
        text = ''
        for page in reader.pages:
            text += page.extract_text()

    # Clean the extracted text
    cleaned_text = clean_text(text)

    return cleaned_text

# Example usage
file_path = 'C1-Annex-2.pdf'  # Replace with your PDF file path
cleaned_text = read_pdf(file_path)
print(cleaned_text)



import spacy

nlp = spacy.load("en_core_web_sm")

def tokenize_text(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    return tokens
print(tokenize_text(cleaned_text))



def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

print(extract_entities(cleaned_text))



import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import json

# Load a new blank spaCy model
nlp = spacy.blank("en")

# Create a DocBin object
db = DocBin()

# Load training data from JSON file
f = open('annotations.json')
TRAIN_DATA = json.load(f)

# Process training data and convert it to spaCy format
for text, annot in tqdm(TRAIN_DATA['annotations']):
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in annot["entities"]:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents
    db.add(doc)

# Save the processed data as a spaCy binary file
db.to_disk("./pvr_training_data.spacy")



nlp_nor = spacy.load("./model-best/model-best") # Include "model-best" folder within the path.
doc = nlp_nor(cleaned_text)  # Input sample text


spacy.displacy.render(doc, style="ent", jupyter=True)  # display in Jupyter




import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

# Assuming 'cleaned_text' is the variable holding the output of read_pdf function
sentence = cleaned_text.split('.')
for i in sentence:
  doc = nlp(i)
  print(f"{'Node (from)-->':<15} {'Relation':^10} {'-->Node (to)':>15}\n")
  for token in doc:
      print("{:<15} {:^10} {:>15}".format(str(token.head.text), str(token.dep_), str(token.text)))
  displacy.render(doc, style='dep')




import stanza
stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

# Call the clean_text function and pass its output to the pipeline
# The variable name was incorrect. Changed 'clean_text' to 'cleaned_text'
doc = nlp(cleaned_text)  # Assuming 'cleaned_text' is the variable containing the raw text

for sent in doc.sentences:
    for word in sent.words:
        print(f'id:{word.id}\nword: {word.text}\nhead id: {word.head}\nhead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel} \n --------------\n', sep='.')