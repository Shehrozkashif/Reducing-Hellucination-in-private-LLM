import re
import json
import os
import PyPDF2
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
from spacy import displacy
import stanza
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

# === Step 1: Clean PDF Text ===
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^A-Za-z0-9\s.,;!?]', '', text)
    return text

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''.join(page.extract_text() for page in reader.pages if page.extract_text())
    return clean_text(text)

# === Step 2: Load and Preprocess PDF ===
file_path = 'data/raw/C1-Annex-2.pdf'  # Change this if needed
cleaned_text = read_pdf(file_path)

print("\n--- Cleaned Text ---\n", cleaned_text[:1000])  # Show only first 1000 characters

# === Step 3: Load spaCy NLP model ===
nlp_spacy = spacy.load("en_core_web_sm")
doc = nlp_spacy(cleaned_text)

# === Step 4: Extract Named Entities ===
def extract_entities(doc):
    return [(ent.text, ent.label_) for ent in doc.ents]

print("\n--- Named Entities ---\n", extract_entities(doc))

# === Step 5: Extract SPO Triplets ===
def extract_knowledge_triplets(doc):
    triplets = []
    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ('nsubj', 'nsubjpass') and token.head.pos_ == 'VERB':
                subject = token.text
                verb = token.head.text
                for child in token.head.children:
                    if child.dep_ in ('dobj', 'pobj', 'attr', 'dative', 'oprd'):
                        obj = child.text
                        triplets.append((subject, verb, obj))
    return triplets

triplets = extract_knowledge_triplets(doc)

print("\n--- Extracted Triplets ---")
for t in triplets:
    print(t)

# === Step 6: Build Knowledge Graph ===
G = nx.DiGraph()

for sub, rel, obj in triplets:
    G.add_node(sub)
    G.add_node(obj)
    G.add_edge(sub, obj, label=rel)

# === Step 7: Visualize Graph with matplotlib ===
pos = nx.spring_layout(G, k=0.5)
plt.figure(figsize=(15, 10))
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, edge_color='gray')
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
plt.title("Knowledge Graph from PDF")
plt.tight_layout()
plt.show()

# === Step 8: Interactive Graph with pyvis ===
net = Network(notebook=False, height="700px", width="100%")

for sub, rel, obj in triplets:
    net.add_node(sub, label=sub)
    net.add_node(obj, label=obj)
    net.add_edge(sub, obj, label=rel)

net.show("knowledge_graph.html")
print("\n✅ Interactive graph saved as 'knowledge_graph.html'")
