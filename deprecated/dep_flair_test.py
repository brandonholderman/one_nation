import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence
from flair.datasets import UD_ENGLISH
from torch.nn.functional import cosine_similarity
from flair.embeddings import TransformerDocumentEmbeddings
from ReadData import ReadData
from progress_bar import run_nerd_bar
# from flair.embeddings import WordEmbeddings
# from flair.models import SequenceTagger
# from flair.trainers import ModelTrainer
# import torch

rd = ReadData()
df = rd._data_reader()
corpus = UD_ENGLISH().downsample(0.1)
print(corpus)


def run_flair():
    sentiment_model = TextClassifier.load('en-sentiment')
    # Create a sentence
    sentence = Sentence(line)
    # Predict sentiment
    sentiment_model.predict(sentence)
    # Access the predicted label
    print(sentence.labels)

    # Load data from CSV
    data_file = "./rdata.csv"
    df = pd.read_csv(data_file)

    # Ensure that the data contains the necessary columns
    if 'Political Lean' not in df.columns or 'Title' not in df.columns:
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
        party = row['Political Lean']
        description = row['Title']
        embeddings[party] = {
            'description': description,
            'embedding': get_embedding(description)
        }

    # Function to calculate similarity between two parties
    def calculate_similarity(emb1, emb2):
        similarity_score = cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        return similarity_score.item()

    # Generate similarity results
    results = []
    parties = list(embeddings.keys())
    for i in range(len(parties)):
        for j in range(i + 1, len(parties)):
            party1 = parties[i]
            party2 = parties[j]
            emb1 = embeddings[party1]['embedding']
            emb2 = embeddings[party2]['embedding']
            desc1 = embeddings[party1]['description']
            desc2 = embeddings[party2]['description']
            
            similarity = calculate_similarity(emb1, emb2)
            
            # Determine the label
            if similarity > 0.7:
                label = 'Strong Positive'
            elif 0.5 < similarity <= 0.7:
                label = 'Moderate Positive'
            elif -0.5 <= similarity <= 0.5:
                label = 'Neutral'
            elif -0.7 <= similarity < -0.5:
                label = 'Moderate Negative'
            else:
                label = 'Strong Negative'

            # Append detailed results
            results.append({
                'Party 1': party1,
                'Party 2': party2,
                'Description 1': desc1,
                'Description 2': desc2,
                'Similarity Score': similarity,
                'Label': label
            })

            # Convert results to DataFrame
            results_df = pd.DataFrame(results)

            # Highlight rows with positive scores
            results_df['Highlight'] = results_df['Label'].apply(lambda x: 'YES' if x == 'Positive' else 'NO')

            # Save the results to a CSV file
            output_file = './comparison.csv'
            results_df.to_csv(output_file, index=False)

            print(f"Detailed similarity analysis completed. Results saved to {output_file}.")


for key, line in df.items():
    if key >= 3:
        run_nerd_bar()
        print('End of Test Run')
        break
    else:
        run_flair()
