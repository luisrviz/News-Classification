import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import nltk
nltk.download('stopwords')


class CleanTextNn:

    @staticmethod
    def process_text(raw_text):
        # Remove punctuation symbols
        text = re.sub("[^\w\s]", "", raw_text)

        # Consider only characters
        text = re.sub("[^a-zA-Z]", " ", text)

        # Remove white spaces
        text = text.strip()

        # Convert all characters to lower cases
        words = text.lower().split()

        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized = [wordnet_lemmatizer.lemmatize(word) for word in words]

        # Remove stopwords
        stops = set(stopwords.words("english"))
        not_stop_words = [w for w in lemmatized if not w in stops]

        return " ".join(not_stop_words)

    @staticmethod
    def datacleaning(text):
        whitespace = re.compile(r"\s+")
        user = re.compile(r"(?i)@[a-z0-9_]+")
        text = whitespace.sub(' ', text)
        text = user.sub('', text)
        text = re.sub(r"\[[^()]*\]","", text)
        text = re.sub("\d+", "", text)
        text = re.sub(r'[^\w\s]','',text)
        text = re.sub(r"(?:@\S*|#\S*|http(?=.*://)\S*)", "", text)
        text = text.lower()
        
        # removing stop-words
        stops = set(stopwords.words("english"))
        not_stop_words = [w for w in text if not w in stops]

        
        # word lemmatization
        sentence = []
        for word in not_stop_words:
            lemmatizer = WordNetLemmatizer()
            sentence.append(lemmatizer.lemmatize(word,'v'))
            
        return ' '.join(sentence)     

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        X["clean_text"] = X["text"].apply(lambda x: self.process_text(x))
        return X


class Tokenization:

    def __init__(self):
        self.tokenizer = Tokenizer(num_words=100000, oov_token='<00V>')

    def fit(self, X, y=None):
        self.tokenizer.fit_on_texts(X.clean_text)
        return self

    def transform(self, X):
        sequences = self.tokenizer.texts_to_sequences(X.clean_text)
        text_embedding_features = pad_sequences(sequences, maxlen=100)
        return text_embedding_features
