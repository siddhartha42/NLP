from util import *

# Add your import statements here


class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = []
		current_sentence = ""
		
		for char in text:
			current_sentence += char
			if char in ['.', '!', '?']:
				segmentedText.append(current_sentence.strip())
				current_sentence = ""
				
		# Add the last sentence if it's not empty
		if current_sentence.strip():
			segmentedText.append(current_sentence.strip())

		return segmentedText

	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""
		
		# Use the Punkt tokenizer for sentence segmentation
		segmentedText = sent_tokenize(text)

		return segmentedText