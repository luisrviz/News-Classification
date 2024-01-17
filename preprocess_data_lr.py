import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
nltk.download('wordnet')
nltk.download('stopwords')


class CleanTextLr:

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

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        X["clean_text"] = X["text"].apply(lambda x: self.process_text(x))
        return X


class TFIDF:

    def fit(self, X, y=None):
        tfidf = TfidfTransformer(norm="l2")
        self.train_text_bigram_tfidf_features = tfidf.fit(X)
        return self

    def transform(self, X):
        return self.train_text_bigram_tfidf_features.transform(X)


class Vectorizer:

    def fit(self, X, y=None):
        bigram_count_vectorizer = CountVectorizer(analyzer="word", ngram_range=(1, 2))
        self.bigram_count_vectorizer = bigram_count_vectorizer.fit(X.clean_text)
        return self

    def transform(self, X):
        return self.bigram_count_vectorizer.transform(X.clean_text)
