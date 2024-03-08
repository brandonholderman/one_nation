from flair.models import TextClassifier
from flair.data import Sentence
from read_data import ReadData

rd = ReadData()
df = rd.data_reader()

for line in df:
    rd.counter_func(15)

    # Load pre-trained sentiment analysis model
    sentiment_model = TextClassifier.load('en-sentiment')
    # Create a sentence
    sentence = Sentence(line)
    # Predict sentiment
    sentiment_model.predict(sentence)
    # Access the predicted label
    print(sentence.labels)



# Limit amount of lines analyzed (range statement)
# def reverse(data):
#     for index in range(len(data)-1, -1, -1):
#         yield data[index]