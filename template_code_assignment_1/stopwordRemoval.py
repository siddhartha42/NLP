from util import *

# Add your import statements here


class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = []

		#get the set of nltk english stopwords 
		stop_words = set(stopwords.words("english")) 

		for sentence in text:
			#adding the words that are not in stop_words to filtered_sentence
			filtered_sentence = [word for word in sentence if word.lower() not in stop_words]
			stopwordRemovedText.append(filtered_sentence)

		return stopwordRemovedText




	