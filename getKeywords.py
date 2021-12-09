from rake_nltk import Rake # https://pypi.org/project/rake-nltk/
import nltk

nltk.download('stopwords')
nltk.download('punkt')

def getKeywords(text):
  # Initialize Rake
  # Uses stopwords for english from NLTK, and all puntuation characters by default
  rake = Rake()
  
  # Extracting keywords from text
  rake.extract_keywords_from_text(text)
  
  # Get dictionary with keywords and scores
  scores = rake.get_word_degrees()
  
  # Return new keywords as list, ignoring scores
  # Obs.: we could change this, selecting only the words with higher score
  return(list(scores.keys()))
