from util import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.vectorizer = TfidfVectorizer()

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable
		"""
		# Flatten the documents list to form the corpus
		corpus = [' '.join([' '.join(sentence) for sentence in doc]) for doc in docs]

		# Generate the TF-IDF matrix
		tfidf_matrix = self.vectorizer.fit_transform(corpus)

		# Build the index as a dictionary mapping from document IDs to TF-IDF vectors
		self.index = dict(zip(docIDs, tfidf_matrix.toarray()))

	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query
		"""
		doc_IDs_ordered = []

		# Flatten the queries list
		queries = [' '.join([' '.join(sentence) for sentence in query]) for query in queries]

		# Transform the query to its TF-IDF representation
		query_tfidf = self.vectorizer.transform(queries)

		# Calculate the cosine similarity of every document with respect to the query
		for query in query_tfidf:
			similarity_scores = {docID: cosine_similarity(query, tfidf_vector.reshape(1, -1)) for docID, tfidf_vector in self.index.items()}

			# Sort the document IDs according to their cosine similarity
			sorted_doc_IDs = sorted(similarity_scores.keys(), key=lambda x: similarity_scores[x], reverse=True)

			doc_IDs_ordered.append(sorted_doc_IDs)

		return doc_IDs_ordered
