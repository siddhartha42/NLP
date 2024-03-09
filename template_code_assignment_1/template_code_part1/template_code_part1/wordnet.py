import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

def print_synset_definitions(word):
    synsets = wordnet.synsets(word)
    print(f"Synsets for '{word}':")
    for synset in synsets:
        print(f" - {synset.name()} : {synset.definition()}")

def path_similarity(word1, word2):
    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)

    max_similarity = 0
    for synset1 in synsets1:
        for synset2 in synsets2:
            similarity = synset1.path_similarity(synset2)
            if similarity is not None and similarity > max_similarity:
                max_similarity = similarity

    return max_similarity

# List of words
words = ['progress', 'advance']

# Print synsets for each word
for word in words:
    print_synset_definitions(word)

# Estimate path-based similarity between 'advance' and 'progress'
similarity = path_similarity('advance', 'progress')
print(f"Path-based similarity between 'advance' and 'progress': {similarity}")
