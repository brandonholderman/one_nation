import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence
from flair.datasets import UD_ENGLISH
from torch.nn.functional import cosine_similarity
from flair.embeddings import TransformerDocumentEmbeddings
from ReadData import ReadData
from progress_bar import run_nerd_bar


# Load data
rd = ReadData()
df = rd._data_reader()
corpus = UD_ENGLISH().downsample(0.1)
print(corpus)

def run_flair():
    # Load Flair's sentiment analysis model
    sentiment_model = TextClassifier.load('en-sentiment')

    # Define input and output file paths
    data_file = "./rdata.csv"
    output_file = "./sentiment_analysis.csv"
    
    # df = pd.read_csv(data_file)


    # Ensure that the data contains the necessary columns
    required_columns = {'Political Lean', 'Title', 'Subreddit'}
    if not required_columns.issubset(df.columns):
        raise ValueError("The dataset must contain 'Political Lean', 'Title', and 'Subreddit' columns.")

    # Clear output file before writing new results
    open(output_file, 'w').close()


    # Function to analyze sentiment of a given text
    def analyze_sentiment(text):
        sentence = Sentence(text)
        sentiment_model.predict(sentence)
        score = sentence.labels[0].score  # Confidence score of sentiment
        label = sentence.labels[0].value  # Positive or Negative

        # Convert raw sentiment label into a more detailed classification
        if label == "POSITIVE":
            if score > 0.75:
                return score, "Strongly Positive"
            elif score > 0.5:
                return score, "Moderately Positive"
            else:
                return score, "Slightly Positive"
        else:  # NEGATIVE
            if score > 0.75:
                return score, "Strongly Negative"
            elif score > 0.5:
                return score, "Moderately Negative"
            else:
                return score, "Slightly Negative"

    # Process each row and analyze sentiment
    results = []
    for index, row in df.iterrows():
        title = row['Title']
        political_lean = row['Political Lean']
        subreddit = row['Subreddit']

        sentiment_score, sentiment_label = analyze_sentiment(title)

        # Store results
        results.append({
            'Title': title,
            'Political Lean': political_lean,
            'Subreddit': subreddit,
            'Sentiment Score': sentiment_score,
            'Sentiment Label': sentiment_label
        })

    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    print(f"Sentiment analysis completed. Results saved to {output_file}.")


for key, line in df.items():
    if key >= 3:
        run_nerd_bar()
        print('End of Test Run')
        break
    else:
        run_flair()