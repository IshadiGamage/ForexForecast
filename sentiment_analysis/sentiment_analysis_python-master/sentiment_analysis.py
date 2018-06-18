import re, math, collections, itertools, os
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
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
def evaluate_features(feature_select):
    #these variables contains the output of our feature selection mechanism
    posFeatures = []
    negFeatures = []
    conFeatures = []
    intFeatures = []
    litFeatures = []
    modFeatures = []
    supFeatures = []
    uncFeatures = []
    newsFeatures = []
    # http://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
    # breaks up the sentences into lists of individual words (as selected by the input mechanism) and appends 'pos' or 'neg' after each list
    with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
        for i in posSentences:
            posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            posWords = [feature_select(posWords), 'pos']
            posFeatures.append(posWords)
    with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
        for i in negSentences:
            negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            negWords = [feature_select(negWords), 'neg']
            negFeatures.append(negWords)
    with open(RT_POLARITY_CON_FILE, 'r') as conSentences:
        for i in conSentences:
            conWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            conWords = [feature_select(conWords), 'con']
            conFeatures.append(conWords)


        # for i in newsSentences:
        #     newsWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        #     newsWords = [feature_select(newsWords), 'news']
        #     newsFeatures.append(newsWords)
    #print(newsFeatures)
    #separates the data into training and testing data for a Naive Bayes classifier
    # selects 3/4 of the features to be used for training and 1/4 to be used for testing
    posCutoff = int(math.floor(len(posFeatures) * 3 / 4))
    negCutoff = int(math.floor(len(negFeatures) * 3 / 4))
    conCutoff = int(math.floor(len(conFeatures) * 3 / 4))

    #trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff] + conFeatures[:conCutoff]
    #testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:] + conFeatures[conCutoff:]
    trainFeatures = posFeatures + negFeatures + conFeatures
    print(trainFeatures)
    with open(RT_POLARITY_NEWS_FILE, 'r') as newsSentences:
        for test_sentence in newsSentences:
            # Tokenize the line.
            doc = nltk.word_tokenize(test_sentence.lower())
            featurized_doc = {i: (i in doc) for i in trainFeatures}
            # tagged_label = classifier.classify(featurized_doc)
            print(doc)
    testFeatures = featurized_doc
    print(trainFeatures)

    # trains a Naive Bayes Classifier
    classifier = NaiveBayesClassifier.train(trainFeatures)

    # initiates referenceSets and testSets
    referenceSets = collections.defaultdict(set) #will contain the actual values for the testing data
    testSets = collections.defaultdict(set) #will contain the predicted output

    # puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
    for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)

        # print(predicted)
        testSets[predicted].add(i)


    # prints metrics to show how well the feature selection did
    print('train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures)))
    print('accuracy:', nltk.classify.util.accuracy(classifier, testFeatures))
    print('pos precision:', precision(referenceSets['pos'], testSets['pos']))
    print('pos recall:', recall(referenceSets['pos'], testSets['pos']))
    print('neg precision:', precision(referenceSets['neg'], testSets['neg']))
    print('neg recall:', recall(referenceSets['neg'], testSets['neg']))
    print('con precision:', precision(referenceSets['con'], testSets['con']))
    print('con recall:', recall(referenceSets['con'], testSets['con']))
    classifier.show_most_informative_features(10)


# creates a feature selection mechanism that uses all words
def make_full_dict(words):
    return dict([(word, True) for word in words])


# tries using all words as the feature selection mechanism
print('using all words as features')
evaluate_features(make_full_dict)


# scores words based on chi-squared test to show information gain (http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/)
def create_word_scores():
    # creates lists of all positive and negative words
    posWords = []
    negWords = []
    conWords = []
    with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
        for i in posSentences:
            posWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            posWords.append(posWord)
    with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
        for i in negSentences:
            negWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            negWords.append(negWord)
    with open(RT_POLARITY_CON_FILE, 'r') as conSentences:
        for i in conSentences:
            conWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            conWords.append(conWord)
    posWords = list(itertools.chain(*posWords))
    negWords = list(itertools.chain(*negWords))
    conWords = list(itertools.chain(*conWords))

    # build frequency distibution of all words and then frequency distributions of words within positive and negative labels
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posWords:
        word_fd[word.lower()] += 1
        cond_word_fd['pos'][word.lower()] += 1
    for word in negWords:
        word_fd[word.lower()] += 1
        cond_word_fd['neg'][word.lower()] += 1
    for word in conWords:
        word_fd[word.lower()] += 1
        cond_word_fd['con'][word.lower()] += 1
    # finds the number of positive and negative words, as well as the total number of words
    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    con_word_count = cond_word_fd['con'].N()
    total_word_count = pos_word_count + neg_word_count + con_word_count

    # builds dictionary of word scores based on chi-squared test
    word_scores = {}
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        con_score = BigramAssocMeasures.chi_sq(cond_word_fd['con'][word], (freq, con_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score + con_score

    return word_scores


# finds word scores
word_scores = create_word_scores()


# finds the best 'number' words based on word scores
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.items(), key=lambda w_s: w_s[0], reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words


# creates feature selection mechanism that only uses best words
def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])


# numbers of features to select
numbers_to_test = [15000]
# tries the best_word_features mechanism with each of the numbers_to_test of features
for num in numbers_to_test:
    print('evaluating best %d word features' % (num))
    best_words = find_best_words(word_scores, num)
    evaluate_features(best_word_features)


