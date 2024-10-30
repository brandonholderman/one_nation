from torch.nn.functional import cosine_similarity
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
import pandas as pd
# import torch

# Load data from CSV
data_file = '/mnt/data/rdata.csv'
df = pd.read_csv(data_file)

# Ensure that the data contains the necessary columns
if 'Party' not in df.columns or 'Description' not in df.columns:
    raise ValueError("The dataset must contain 'Party' and 'Description' columns.")

# Initialize transformer embeddings (using a base model like 'distilbert-base-uncased')
embedding_model = TransformerDocumentEmbeddings('distilbert-base-uncased')


# Generate embeddings for each party's description
def get_embedding(text):
    sentence = Sentence(text)
    embedding_model.embed(sentence)
    return sentence.get_embedding()


# Store embeddings in a dictionary for reference
embeddings = {}
for index, row in df.iterrows():
    party = row['Party']
    description = row['Description']
    embeddings[party] = get_embedding(description)


# Function to calculate similarity between two parties
def calculate_similarity(party1, party2):
    emb1 = embeddings[party1]
    emb2 = embeddings[party2]
    similarity_score = cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
    return similarity_score.item()


# Generate similarity results
results = []
parties = list(embeddings.keys())
for i in range(len(parties)):
    for j in range(i + 1, len(parties)):
        party1 = parties[i]
        party2 = parties[j]
        similarity = calculate_similarity(party1, party2)
        
        # Label the similarity
        if similarity > 0.5:
            label = 'Positive'
        elif similarity < -0.5:
            label = 'Negative'
        else:
            label = 'Neutral'
        
        # Append to results
        results.append({
            'Party 1': party1,
            'Party 2': party2,
            'Similarity Score': similarity,
            'Label': label
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save the results to a CSV file
output_file = '/mnt/data/political_party_similarity.csv'
results_df.to_csv(output_file, index=False)

print(f"Similarity analysis completed. Results saved to {output_file}.")
