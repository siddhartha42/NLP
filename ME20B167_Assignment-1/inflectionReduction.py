from util import *

class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = []

		stemmer = PorterStemmer() #initialize porter stemmer object

		for sentence in text:
			#iteration over all tokens and stemming
			stemmed_sentence = [stemmer.stem(token) for token in sentence] 
			reducedText.append(stemmed_sentence)
		
		return reducedText


