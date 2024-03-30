from util import *

# Add your import statements here

class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []
		
		for sentence in text:
			# Split each sentence into tokens using spaces, tabs and newlines
			tokens = re.split(r'\s+|\t+|\n+', sentence) 
			tokenizedText.append(tokens)
		
		return tokenizedText

	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		for sentence in text:
			#try - except for easy debugging
			try:
				tokens = word_tokenize(sentence) #using NLTK's word tokenize function
			except Exception as e:
				raise Exception("Error occurred during tokenization:", e) 
			tokenizedText.append(tokens)

		return tokenizedText