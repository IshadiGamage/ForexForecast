from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
import re, math, collections, itertools, os
import nltk, nltk.classify.util, nltk.metrics
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import precision
from nltk.metrics import recall

POLARITY_DATA_DIR = os.path.join('polarityData', 'rt-polaritydata')
RT_POLARITY_NEG_FILE = os.path.join(POLARITY_DATA_DIR, 'Negative.txt')
RT_POLARITY_POS_FILE = os.path.join(POLARITY_DATA_DIR, 'Positive.txt')
RT_POLARITY_CON_FILE = os.path.join(POLARITY_DATA_DIR, 'Constraining.txt')
RT_POLARITY_INT_FILE = os.path.join(POLARITY_DATA_DIR, 'Interesting.txt')
RT_POLARITY_LIT_FILE = os.path.join(POLARITY_DATA_DIR, 'Litigious.txt')
RT_POLARITY_MOD_FILE = os.path.join(POLARITY_DATA_DIR, 'Modal.txt')
RT_POLARITY_SUP_FILE = os.path.join(POLARITY_DATA_DIR, 'Superfluous.txt')
RT_POLARITY_UNC_FILE = os.path.join(POLARITY_DATA_DIR, 'Uncertainty.txt')
RT_POLARITY_NEWS_FILE = os.path.join(POLARITY_DATA_DIR, 'news.txt')
# this function takes a feature selection mechanism and returns its performance in a variety of metrics

#these variables contains the output of our feature selection mechanism
posFeatures = []
negFeatures = []

with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
    for i in posSentences:
        posWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        posWord= (posWord, 'pos')
        posFeatures.append(posWord)
with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
    for i in negSentences:
        negWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        negWord = (negWord, 'neg')
        negFeatures.append(negWord)
print(posFeatures)
trainFeatures = posFeatures + negFeatures
train = [
    ('I love this sandwich.', 'pos'),
    ('This is an amazing place!', 'pos'),
    ('I feel very good about these beers.', 'pos'),
    ('This is my best work.', 'pos'),
    ("What an awesome view", 'pos'),
    ('I do not like this restaurant', 'neg'),
    ('I am tired of this stuff.', 'neg'),
    ("I can't deal with this", 'neg'),
    ('He is my sworn enemy!', 'neg'),
    ('My boss is horrible.', 'neg')
]
test = [
    ('The beer was good.', 'pos'),
    ('I do not enjoy my job', 'neg'),
    ("I ain't feeling dandy today.", 'neg'),
    ("I feel amazing!", 'pos'),
    ('Gary is a friend of mine.', 'pos'),
    ("I can't believe I'm doing this.", 'neg')
]

cl = NaiveBayesClassifier(trainFeatures)
with open(RT_POLARITY_NEWS_FILE, 'r') as content_file:
    content = content_file.read()
# Classify some text
print(cl.classify(content))  # "pos"
print(cl.classify("ACCOMPLISHING ACHIEVED ADVANTAGES"))   # "neg"


print("Accuracy: {0}".format(cl.accuracy(test)))

# Show 5 most informative features
cl.show_informative_features(5)