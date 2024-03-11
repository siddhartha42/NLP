from util import *

# Add your import statements here

class SpellCheck():
     
    def __init__(self, vocabulary_vectors):
        self.vocabulary_vectors = vocabulary_vectors
        self.unique_tokens = set(vocabulary_vectors.keys())
     
    def errors(self, text):
        """
        Spell checking using vocabulary built from documents in cranfield dataset

        Parameters
        ----------
        text : list
            A list of typos

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
            representing the top 5 candidate corrections
        """

        corrections = []

        for typos in text:
            for typo in typos:
                candidates = self.find_candidate_corrections(typo)
                corrections.append(candidates)
        return corrections
     
    def find_candidate_corrections(self, typo):
        """
        Find top 5 candidate corrections for a typo

        Parameters
        ----------
        typo : str
            The typo to find corrections for

        Returns
        -------
        list
            A list of tuples, each containing a token and its similarity score
        """
        candidates = []

        typo_vector = word_to_vector(typo)

        for token in self.unique_tokens:
            token_vector = self.vocabulary_vectors.get(token, [0] * len(typo_vector))
            sim = cosine_similarity(typo_vector, token_vector)
            candidates.append((token, sim))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:5]




	