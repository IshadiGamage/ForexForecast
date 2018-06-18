import nltk
import random
import pickle
from nltk.corpus import movie_reviews
from os.path import exists
from nltk.classify import apply_features
from nltk.tokenize import word_tokenize, sent_tokenize

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())
print(word_features)

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]
numtrain = int(len(documents) * 90 / 100)
training_set = apply_features(find_features, documents[:numtrain])
testing_set = apply_features(find_features, documents[numtrain:])

classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier.show_most_informative_features(15)

Example_Text = " avoids annual conveys vocal thematic doubts fascination slip avoids outstanding thematic astounding seamless"

doc = word_tokenize(Example_Text.lower())
featurized_doc = {i:(i in doc) for i in word_features} 
tagged_label = classifier.classify(featurized_doc)
print(tagged_label)