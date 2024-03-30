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
			current_sentence += char #adding a character to the sentence until a ., !, ? appears
			if char in ['.', '!', '?']:
				segmentedText.append(current_sentence.strip()) #adding the sentence to the list
				current_sentence = "" #reinitializing sentence to empty string
				
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
		
		#try - except for easy debugging in case of errors
		try:
			segmented_text = sent_tokenize(text) #using punkt for sentence segmentation
		except Exception as e:
			raise Exception("Error occurred during sentence segmentation:", e) 
		
		return segmented_text