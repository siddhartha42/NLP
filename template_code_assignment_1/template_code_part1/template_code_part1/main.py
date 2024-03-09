from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from spellCheck import SpellCheck
from util import *

import argparse
import json
from sys import version_info

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")


class SearchEngine:

	def __init__(self, args):
		self.args = args
		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.spellChecker = None
		self.inflectionReducer = InflectionReduction()
		self.stopwordRemover = StopwordRemoval()

	def segmentSentences(self, text):
		"""
		Return the required sentence segmenter
		"""
		if self.args.segmenter == "naive":
			return self.sentenceSegmenter.naive(text)
		elif self.args.segmenter == "punkt":
			return self.sentenceSegmenter.punkt(text)

	def tokenize(self, text):
		"""
		Return the required tokenizer
		"""
		if self.args.tokenizer == "naive":
			return self.tokenizer.naive(text)
		elif self.args.tokenizer == "ptb":
			return self.tokenizer.pennTreeBank(text)
		
	def spellCheck(self, text):
		"""
		Return the errors 
		"""
		return self.spellChecker.errors(text)

	def reduceInflection(self, text):
		"""
		Return the required stemmer/lemmatizer
		"""
		return self.inflectionReducer.reduce(text)

	def removeStopwords(self, text):
		"""
		Return the required stopword remover
		"""
		return self.stopwordRemover.fromList(text)


	def preprocessQueries(self, queries):
		"""
		Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
		"""

		# Segment queries
		segmentedQueries = []
		for query in queries:
			segmentedQuery = self.segmentSentences(query)
			segmentedQueries.append(segmentedQuery)
		json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
		
		# Tokenize queries
		tokenizedQueries = []
		for query in segmentedQueries:
			tokenizedQuery = self.tokenize(query)
			tokenizedQueries.append(tokenizedQuery)
		json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
		
		# Stem/Lemmatize queries
		reducedQueries = []
		for query in tokenizedQueries:
			reducedQuery = self.reduceInflection(query)
			reducedQueries.append(reducedQuery)
		json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
		
		# Remove stopwords from queries
		stopwordRemovedQueries = []
		for query in reducedQueries:
			stopwordRemovedQuery = self.removeStopwords(query)
			stopwordRemovedQueries.append(stopwordRemovedQuery)
		json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))
		
		if (args.custom):
			# Spell check queries
			spellCheckedQueries = []
			closest_corrections = []
			for query in stopwordRemovedQueries:
				spellCheckedQuery = self.spellCheck(query)
				spellCheckedQueries.append(spellCheckedQuery)
				for q in query:
					for qy in q:
						print(qy)
						closest_correction = None
						min_distance = float('inf')
						for spellCheckedQuery in spellCheckedQueries:
							for corrected_word in spellCheckedQuery:
								for words in corrected_word:
									print(words)
									distance = editDistance(qy, words[0])
									if distance < min_distance:
										min_distance = distance
										closest_correction = words[0]
						closest_corrections.append((closest_correction, min_distance))

			json.dump(spellCheckedQueries, open(self.args.out_folder + "spell_checked_queries.txt", 'w'))

			# Dump the closest corrections to a file
			json.dump(closest_corrections, open(self.args.out_folder + "closest_corrections.txt", 'w'))

		preprocessedQueries = stopwordRemovedQueries
		return preprocessedQueries

	def preprocessDocs(self, docs):
		"""
		Preprocess the documents
		"""
		
		# Segment docs
		segmentedDocs = []
		for doc in docs:
			segmentedDoc = self.segmentSentences(doc)
			segmentedDocs.append(segmentedDoc)
		json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
		
		# Tokenize docs
		tokenizedDocs = []
		for doc in segmentedDocs:
			tokenizedDoc = self.tokenize(doc)
			tokenizedDocs.append(tokenizedDoc)
		json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
		
		# Stem/Lemmatize docs
		reducedDocs = []
		for doc in tokenizedDocs:
			reducedDoc = self.reduceInflection(doc)
			reducedDocs.append(reducedDoc)
		json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
		
		# Remove stopwords from docs
		stopwordRemovedDocs = []
		for doc in reducedDocs:
			stopwordRemovedDoc = self.removeStopwords(doc)
			stopwordRemovedDocs.append(stopwordRemovedDoc)
		json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))
		
		if(args.custom):
			# Build vocabulary from processed docs
			vocabulary_vectors = buildVocabulary(stopwordRemovedDocs)
			# Initialize SpellCheck with the vocabulary
			self.spellChecker = SpellCheck(vocabulary_vectors)

		preprocessedDocs = stopwordRemovedDocs
		return preprocessedDocs


	def evaluateDataset(self):
		"""
		Evaluate document-query relevances for all document-query pairs
		"""

		# Read queries
		queries_json = json.load(open(args.dataset + "\\cran_queries.json", 'r'))[:]
		queries = [item["query"] for item in queries_json]
		# Process queries 
		processedQueries = self.preprocessQueries(queries)

		# Read documents
		docs_json = json.load(open(args.dataset + "\\cran_docs.json", 'r'))[:]
		docs = [item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		nltk.download('stopwords')
		nltk.download('punkt')

		# Remaning code will be added later


	def handleCustomQuery(self):
		"""
		Take a custom query as input and return relevances with all documents
		"""

		#Get query
		print("Enter query below")
		query = input()

		# Read documents
		docs_json = json.load(open(args.dataset + "\\cran_docs.json", 'r'))[:]
		docs = [item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Process query
		processedQuery = self.preprocessQueries([query])[0]

		# Download the Punkt tokenizer if not already downloaded
		nltk.download('punkt')
		nltk.download('stopwords')
		# Remaning code will be added later


if __name__ == "__main__":

	# Create an argument parser
	parser = argparse.ArgumentParser(description='main.py')

	# Tunable parameters as external arguments
	parser.add_argument('-dataset', default = "cranfield/", 
						help = "Path to the dataset folder")
	parser.add_argument('-out_folder', default = "output/", 
						help = "Path to output folder")
	parser.add_argument('-segmenter', default = "punkt",
	                    help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('-tokenizer',  default = "ptb",
	                    help = "Tokenizer Type [naive|ptb]")
	parser.add_argument('-custom', action = "store_true", 
						help = "Take custom query as input")
	
	# Parse the input arguments
	args = parser.parse_args()

	# Create an instance of the Search Engine
	searchEngine = SearchEngine(args)

	# Either handle query from user or evaluate on the complete dataset 
	if args.custom:
		searchEngine.handleCustomQuery()
	else:
		searchEngine.evaluateDataset()
