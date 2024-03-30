'''
Code to create a custom stopword list from cranfield dataset 
and visually compare it with the nltk stopword list

EDIT THE PATH TO DATASET AND RUN THIS FILE SEPARATELY TO VIEW GRAPH IN DOC
'''

import matplotlib.pyplot as plt
import json
import nltk
from nltk.corpus import stopwords
from collections import Counter

#loading the dataset 
def load_corpus_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    corpus = [document['body'] for document in data]
    return corpus

# Step 1: Tokenization
def tokenize_corpus(corpus):
    tokens = []
    for document in corpus:
        tokens.extend(nltk.word_tokenize(document.lower()))  # Tokenize and convert to lowercase
    return tokens

# Step 2: Term Frequency Calculation
def calculate_term_frequency(tokens):
    return Counter(tokens)

# Step 3: Selecting Stopwords based on threshold
def select_stopwords(term_freq, threshold):
    return [word for word, freq in term_freq.items() if freq > threshold]

# Load corpus from JSON file (EDIT THE PATH TO DATASET HERE AND RUN THE FILE SEPARATELY)
corpus_file = r"C:\Users\Siddhartha\OneDrive\Desktop\Final Sem\NLP\template_code_assignment_1\cranfield\cran_docs.json"
corpus = load_corpus_from_json(corpus_file)

# Tokenization
tokens = tokenize_corpus(corpus)

# Term Frequency Calculation
term_freq = calculate_term_frequency(tokens)

# NLTK stopwords
nltk_stopwords = stopwords.words('english')

# Initialize lists to store the number of common words and threshold values
common_words_count = []
threshold_values = range(1, 50, 1) 

# Calculate common words count for each threshold value
for threshold in threshold_values:
    # Custom stopwords
    custom_stopwords = select_stopwords(term_freq, threshold)
    
    # Count number of common words
    common_count = len(set(custom_stopwords).intersection(nltk_stopwords))
    common_words_count.append(common_count)

# Plot the number of common words against threshold values
plt.plot(threshold_values, common_words_count, marker='o')
plt.title('Number of Common Words vs. Threshold Value')
plt.xlabel('Threshold Value')
plt.ylabel('Number of Common Words')
plt.grid(True)
plt.show()

# Turn off interactive mode (to end code execution after the plot window is closed)
plt.ioff()
