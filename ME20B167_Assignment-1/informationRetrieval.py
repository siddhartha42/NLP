# from util import *
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# class InformationRetrieval():

#     def __init__(self):
#         self.index = None
#         self.vectorizer = None
#         self.feature_names = None
#         self.matrix = None

#     def buildIndex(self, docs, docIDs):
#         """
#         Builds the document index in terms of the document
#         IDs and stores it in the 'index' class variable

#         Parameters
#         ----------
#         docs : list
#             A list of lists of lists where each sub-list is
#             a document and each sub-sub-list is a sentence of the document
#         docIDs : list
#             A list of integers denoting IDs of the documents
#         Returns
#         -------
#         None
#         """
    
#         doc_strings = []
#         for doc in docs:
#             doc_list = []
#             for sentence in doc:
#                 sentence_string = ' '.join(sentence)
#                 doc_list.append(sentence_string)
#             doc_strings.append(' '.join(doc_list))
                    

#         tfidf_vectorizer = TfidfVectorizer()
#         tfidf_matrix = tfidf_vectorizer.fit_transform(doc_strings)
#         self.matrix = tfidf_matrix
#         feature_names = tfidf_vectorizer.get_feature_names_out()
#         self.feature_names = feature_names
#         self.vectorizer = tfidf_vectorizer
#         tfidf_vectors = []
#         for doc_index in range(len(doc_strings)):
#             doc_vector = tfidf_matrix[doc_index].toarray().flatten()
#             tfidf_vectors.append(dict(zip(feature_names, doc_vector)))
           
#         # Map each document vector to its docID
#         index = {}
#         for docID, tfidf_vector in zip(docIDs, tfidf_vectors):
#             index[docID] = tfidf_vector
#         self.index = index
        

#     def rank(self, queries):
#         """
#         Rank the documents according to relevance for each query

#         Parameters
#         ----------
#         queries : list
#             A list of lists of lists where each sub-list is a query and
#             each sub-sub-list is a sentence of the query
#         Returns
#         -------
#         list
#             A list of lists of integers where the ith sub-list is a list of IDs
#             of documents in their predicted order of relevance to the ith query
#         """

#         doc_IDs_ordered = []

#         # Convert queries into TF-IDF matrix using the same vectorizer
#         query_strings = [' '.join([' '.join(sentence) for sentence in query]) for query in queries]
#         query_tfidf_matrix = self.vectorizer.transform(query_strings)

#         # Loop through each query
#         for query_vector in query_tfidf_matrix:
#             query_vector = query_vector.toarray().flatten()
            
#             # Calculate cosine similarity between query and documents
#             cosine_similarities = cosine_similarity([query_vector], self.matrix)
#             # Sort documents by relevance
#             sorted_doc_ids = [docID for docID, _ in sorted(zip(self.index.keys(), cosine_similarities[0]), key=lambda x: x[1], reverse=True)]
#             doc_IDs_ordered.append(sorted_doc_ids)

#         return doc_IDs_ordered




# LSA

from util import *
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

class InformationRetrieval():
    def __init__(self):
        self.index = None
        self.svd = None
        self.vectorizer = None

    def buildIndex(self, docs, docIDs, n_components=100):
        """
        Builds the document index using Latent Semantic Analysis (LSA)

        Parameters
        ----------
        docs : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        docIDs : list
            A list of integers denoting IDs of the documents
        n_components : int, optional
            Number of components to keep in LSA, by default 100
        Returns
        -------
        None
        """
        doc_strings = []
        for doc in docs:
            doc_list = []
            for sentence in doc:
                sentence_string = ' '.join(sentence)
                doc_list.append(sentence_string)
            doc_strings.append(' '.join(doc_list))

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(doc_strings)
        self.matrix = tfidf_matrix
        self.feature_names = tfidf_vectorizer.get_feature_names_out()
        self.vectorizer = tfidf_vectorizer
        
        # LSA pipeline: TF-IDF -> SVD -> Normalizer
        lsa_pipeline = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components), Normalizer(copy=False))
        lsa_matrix = lsa_pipeline.fit_transform(doc_strings)
        
        # Map each document vector to its docID
        self.index = {docID: lsa_matrix[i] for i, docID in enumerate(docIDs)}
        self.svd = lsa_pipeline.named_steps['truncatedsvd']

    def rank(self, queries):
        """
        Rank the documents according to relevance for each query using LSA

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

        # Transform queries using TF-IDF vectorizer
        query_strings = [' '.join([' '.join(sentence) for sentence in query]) for query in queries]
        query_tfidf_matrix = self.vectorizer.transform(query_strings)

        # Transform TF-IDF matrix using LSA
        query_vectors = self.svd.transform(query_tfidf_matrix)

        # Loop through each query
        for query_vector in query_vectors:
            # Calculate cosine similarity between query and documents
            cosine_similarities = cosine_similarity([query_vector], list(self.index.values()))
            # Sort documents by relevance
            sorted_doc_ids = [docID for docID, _ in sorted(zip(self.index.keys(), cosine_similarities[0]), key=lambda x: x[1], reverse=True)]
            doc_IDs_ordered.append(sorted_doc_ids)

        return doc_IDs_ordered
    

