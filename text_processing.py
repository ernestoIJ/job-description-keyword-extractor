import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords from NLTK
nltk.download('stopwords') # Stop words like 'the', 'and', and 'I', because they don't provide meaniful information
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """
    Preprocesses the input text by cleaning and normalizing it. This includes removing carriage returns,
    new lines, punctuation, and numbers, converting to lowercase, tokenizing, and removing stopwords and non-alphabetic
    characters.

    Args:
        text (str): A string containing the text to be processed.

    Returns:
        str: The processed text, which is cleaned, tokenized, and stripped of stopwords and non-alphabetic
        characters, concatenated back into a single string.
    """

    # remove \r and \n
    text = re.sub(r'[\r\n]+', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)

    # tokenize
    words = word_tokenize(text)

    # Remove stopwords and non-alphabetic characters
    words = [word for word in words if word.isalpha() and word not in stop_words]

    return ' '.join(words)


def extract_keywords(texts: list, num_keywords: int=10) -> list:
    """
    Extracts keywords from a list containing texts (str) using the TF-IDF (Term Frequency - Inverse Document Frequency) method.
    This function initialized a TF-IDF vectorizer, fits it to the provided texts, and extracts the top features (keywords)
    based on their TF-IDF scores.

    Args:
        texts (list): A list of preprocessed text (str) documents from which to extract keywords.
        num_keywords (int): The number of top keywords to extract from each document. Defaults to 10.

    Returns:
        list: A list of the top `num_keywords` extracted as the most relevant keywords from the texts.
    """

    # Initialize a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=num_keywords, stop_words='english')

    # Fit the model and transform the data
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    # Extract the features names, which are the keywords
    feature_names = tfidf_vectorizer.get_feature_names_out()

    return feature_names.tolist()