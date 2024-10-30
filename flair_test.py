import torch
import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence
from flair.datasets import UD_ENGLISH
from flair.embeddings import WordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.nn.functional import cosine_similarity
from flair.embeddings import TransformerDocumentEmbeddings
from ReadData import ReadData
from progress_bar import run_nerd_bar

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
output_file = './political_party_similarity.csv'
results_df.to_csv(output_file, index=False)

print(f"Similarity analysis completed. Results saved to {output_file}.")

for line in df.items():
    if line >= 10:
        run_nerd_bar()
        print('End of Test Run')
        break
    else:
        run_flair()


# def run_trainer():
#     # 2. what label do we want to predict?
#     label_type = 'upos'

#     # 3. make the label dictionary from the corpus
#     label_dict = corpus.make_label_dictionary(label_type=label_type)
#     print(label_dict)

#     # 4. initialize embeddings
#     embeddings = WordEmbeddings('glove')

#     # 5. initialize sequence tagger
#     model = SequenceTagger(hidden_size=256,
#                             embeddings=embeddings,
#                             tag_dictionary=label_dict,
#                             tag_type=label_type)

#     # 6. initialize trainer
#     trainer = ModelTrainer(model, corpus)

#     # 7. start training
#     trainer.train('resources/taggers/example-upos',
#                 learning_rate=0.1,
#                 mini_batch_size=32,
#                 max_epochs=10)

# def run_trainer():
#     # Initialize training model
#     pretrained_model = SequenceTagger.load('ner')

#     trainer : ModelTrainer = ModelTrainer(pretrained_model, corpus)

#     trainer.fine_tune('resources/taggers/example-ner',
#                     learning_rate=0.1,
#                     mini_batch_size=2,
#                     max_epochs=3)

# for line in df:
# run_trainer()