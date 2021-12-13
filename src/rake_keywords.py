from rake_nltk import Rake # https://pypi.org/project/rake-nltk/
from nltk_resources import download_all_nltk_resources_if_needed  
download_all_nltk_resources_if_needed() # TODO: download only nedeed resources

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

if __name__ == '__main__':
  text = "This is a sample text. It has nothing to do with the keywords extraction."
  print(getKeywords(text))