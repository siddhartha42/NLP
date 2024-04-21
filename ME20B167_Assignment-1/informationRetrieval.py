from util import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class InformationRetrieval():

    def __init__(self):
        self.index = None
        self.vectorizer = None
        self.corpus = None

    def buildIndex(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        docs : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        docIDs : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """
        # Create a corpus of all unique words
        self.corpus = set(word for doc in docs for sentence in doc for word in sentence)

        # Create TF-IDF vectorizer with the corpus
        self.vectorizer = TfidfVectorizer(vocabulary=self.corpus)

        # Transform documents into TF-IDF vector representations
        doc_strings = [' '.join([' '.join(word) for word in doc]) for doc in docs]
        tfidf_matrix = self.vectorizer.fit_transform(doc_strings)
     
        # Map each document vector to its docID
        index = {}
        for docID, tfidf_vector in zip(docIDs, tfidf_matrix):
            index[docID] = tfidf_vector

        self.index = index
        

    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        queries : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query
        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """

        doc_IDs_ordered = []

        # Loop through each query
        for query in queries:
            # Convert query into TF-IDF vector
            query_str = ' '.join([' '.join(sentence) for sentence in query])
            query_vector = self.vectorizer.transform([query_str])

            # Calculate cosine similarity between query and documents
            cosine_similarities = {}
            for docID, doc_vector in self.index.items():
                cosine_similarities[docID] = cosine_similarity(query_vector, doc_vector)

            # Sort documents by relevance
            sorted_doc_ids = [docID for docID, _ in sorted(cosine_similarities.items(), key=lambda x: x[1], reverse=True)]

            doc_IDs_ordered.append(sorted_doc_ids)

        return doc_IDs_ordered
