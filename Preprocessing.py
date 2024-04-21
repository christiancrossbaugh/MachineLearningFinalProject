import pandas as pd

# Function to swap two columns in a DataFrame
def swap_columns(df, col1, col2):
    df[col1], df[col2] = df[col2], df[col1]
    return df

# Read the CSV file into a DataFrame
df = pd.read_csv('/content/AIGeneratedAndLabeled.csv.txt')

# Specify the names of the columns to swap
column1 = 'requirement'
column2 = 'reqLabel'

# Swap the columns
df = swap_columns(df, column1, column2)

# Save the modified DataFrame to a new CSV file
df.to_csv('output.csv', index=False)




#Uses sentence to vector library to iterate over a csv of sentences and return the most common 10 tokens.

import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

nltk.download('sent2vec')
import sent2vec

# Load sent2vec model
model_path = 'path/to/sent2vec_model.bin'  # Replace with the path to your sent2vec model
model = sent2vec.Sent2vecModel()
model.load_model(model_path)

# Function to preprocess sentences
def preprocess(sentence):
    # Tokenize sentence
    tokens = word_tokenize(sentence.lower())
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english') + list(punctuation))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

# Load CSV file
csv_file = "/content/output.csv"  # Replace with the path to your CSV file
df = pd.read_csv(csv_file)

# Combine all sentences into one list
sentences = df['requirement'].tolist()  # Assuming 'requirement' is the column with sentences
all_tokens = []

# Iterate over sentences and preprocess them
for sentence in sentences:
    tokens = preprocess(sentence)
    all_tokens.extend(tokens)

# Count the frequency of each token
token_counts = Counter(all_tokens)

# Get the top 10 most common tokens
top_tokens = token_counts.most_common(10)
print("Top 10 most common tokens:")
for token, count in top_tokens:
    print(f"{token}: {count}")



# Preprocesses CSV to remove punctuation then counts the words and adds new column to csv titled word_count

import pandas as pd
import re
from nltk.tokenize import word_tokenize

# Function to preprocess the sentence and count the number of words
def count_words(sentence):
    # Preprocess the sentence: remove punctuation and extra spaces
    sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    # Tokenize the preprocessed sentence and return the word count
    tokens = word_tokenize(sentence)
    return len(tokens)

# Read CSV file into a DataFrame
df = pd.read_csv('/content/software_req_classified_raw.csv')

# Count words in the 'requirements' column
df['word_count'] = df['requirement'].apply(count_words)

# Export DataFrame with word count as a new column to a new CSV file
df.to_csv('output.csv', index=False)


import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
nltk.download('punkt')
nltk.download('stopwords')

# Load BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to preprocess sentences
def preprocess(sentence):
    # Tokenize sentence
    tokens = word_tokenize(sentence.lower())
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english') + list(punctuation))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

# Load CSV file
csv_file = "output.csv"  # Replace with the path to your CSV file
df = pd.read_csv(csv_file)

# Combine all sentences into one list
sentences = df['requirement'].tolist()  # Assuming 'requirement' is the column with sentences
all_tokens = []

# Iterate over sentences and preprocess them
for sentence in sentences:
    tokens = preprocess(sentence)
    all_tokens.extend(tokens)

# Count the frequency of each token
token_counts = Counter(all_tokens)

# Get the top 10 most common tokens
top_tokens = token_counts.most_common(10)
print("Top 10 most common tokens:")
for token, count in top_tokens:
    print(f"{token}: {count}")