# Add your import statements here
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import json
from collections import Counter
from itertools import product
import math

# Add any utility functions here

import re

def buildVocabulary(docs_list):
    """
    Build the vocabulary from the processed documents

    Parameters
    ----------
    docs : list
        List of processed documents

    Returns
    -------
    dict
        Vocabulary represented as token to vector mappings
    """
    
    tokens = []

    # Extract all alphanumeric tokens from the processed documents
    for docs in docs_list:
        for doc in docs:
            for token in doc:
                if (token.isalnum() and isinstance(token,str)):
                    tokens.append(token)

    # Find unique tokens
    unique_tokens = set(tokens)

    # Initialize an empty dictionary to store vocabulary vectors
    vocabulary_vectors = {}

    # Loop through each unique token to generate its vector representation
    for token in unique_tokens:
        # Generate vector representation for the token
        vector = word_to_vector(token)

        # Store the token and its vector representation in the vocabulary dictionary
        vocabulary_vectors[token] = vector

    # Return the vocabulary dictionary
    return vocabulary_vectors


def word_to_vector(word):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    bigrams = [ a + b for a , b in product(alphabet, repeat=2)]

    vector = [0] * len(bigrams)
    for i in range(len(word) - 1):
        bigram = word[i:i+2]
        if bigram in bigrams:
            vector[bigrams.index(bigram)] += 1

    return vector

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a , b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a**2 for a in vec1))
    magnitude2 = math.sqrt(sum(b**2 for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    else:
        return dot_product / (magnitude1 * magnitude2) 