
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
 
def word_feats(words):
    return dict([(word, True) for word in words])

positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)' ]
negative_vocab = [ 'bad', 'terrible','useless', 'hate', ':(' ]
neutral_vocab = [ 'movie','the','sound','was','is','actors','did','know','words','not' ]
 
positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]
# print(positive_features)

train_set = positive_features + negative_features + neutral_features
# print(train_set)
classifier = NaiveBayesClassifier.train(train_set) 

 
# Predict
neg = 0
pos = 0
neu = 0
sentence = "ABLE ADVANCES ACHIEVES"
sentence = sentence.lower()
words = sentence.split(' ')
# print(words)
# print(word_feats(words))
for word in words:
    classResult = classifier.classify( word_feats(word))
    print(classResult)
    if classResult == 'pos':
        pos = pos + 1
    if classResult == 'neg':
        neg = neg + 1    
    if classResult == 'neu':
        neu = neu + 1
 
print('Positive: ' + str(float(pos)/len(words)))
print('Negative: ' + str(float(neg)/len(words)))
print('Neutral: ' + str(float(neu) /len(words)))



