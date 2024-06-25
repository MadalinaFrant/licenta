import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def clean_text(text):
    pattern = re.compile(r'^[a-zA-Z]+$')

    words = []
    for word in word_tokenize(text.lower()):
        if pattern.match(word):
            words.append(word)

    return words


def get_stopwords():
    stop_words = []
    for lang in stopwords.fileids():
        stop_words += stopwords.words(lang)

    return set(stop_words)


def remove_stopwords(word_list):
    stop_words = get_stopwords()

    words = []
    for word in word_list:
        if word not in stop_words:
            words.append(word)

    return words


def pos_tagger(nltk_tag):

    if nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('R'):
        return wordnet.ADV

    return None


def lemmatize_words(words):
    lemmatizer = WordNetLemmatizer()

    tagged_words = []
    for word, tag in nltk.pos_tag(words):
        tagged_words.append((word, pos_tagger(tag)))

    lemmatized_words = []
    for word, tag in tagged_words:
        if tag is None:
            lemmatized_words.append(word)
        else:        
            lemmatized_words.append(lemmatizer.lemmatize(word, tag))

    return lemmatized_words


def preprocess_text(text):

    word_list = clean_text(text)
    words = remove_stopwords(word_list)
    lemmatized_words = lemmatize_words(words)

    cleaned_text = ' '.join(lemmatized_words)

    return cleaned_text

