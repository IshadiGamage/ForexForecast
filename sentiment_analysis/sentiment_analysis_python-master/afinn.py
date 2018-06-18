#!/usr/bin/python
#
# (originally entered at https://gist.github.com/1035399)
#
# License: GPLv3
#
# To download the AFINN word list do:
# wget http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/6010/zip/imm6010.zip
# unzip imm6010.zip
#
# Note that for pedagogic reasons there is a UNICODE/UTF-8 error in the code.

import math
import re
import sys
import os
from nltk.corpus import stopwords

# AFINN-111 is as of June 2011 the most recent version of AFINN
filenameAFINN = 'AFINN/AFINN-111.txt'
afinn = dict([(w_s[0], int(w_s[1])) for w_s in [
            ws.strip().split('\t') for ws in open(filenameAFINN) ]])
# print(afinn)

# Word splitter pattern
pattern_split = re.compile(r"\W+")

def sentiment(text):
    """
    Returns a float for sentiment strength based on the input text.
    Positive values are positive valence, negative value are negative valence.
    """

    words = pattern_split.split(text.lower())
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    sentiments = [afinn.get(word, 0) for word in filtered_words]
    print (filtered_words)
    if sentiments:
        # How should you weight the individual word sentiments?
        # You could do N, sqrt(N) or 1 for example. Here I use sqrt(N)
        sentiment = float(sum(sentiments))/math.sqrt(len(sentiments))

    else:
        sentiment = 0
    return sentiment

POLARITY_DATA_DIR = os.path.join('polarityData', 'rt-polaritydata')
RT_POLARITY_NEWS_FILE = os.path.join(POLARITY_DATA_DIR, 'news.txt')

with open(RT_POLARITY_NEWS_FILE, 'r') as newsSentences:
    for i in newsSentences:
        words=re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        text = newsSentences.read().replace('\n', '')
        print(sentiment(text))
if __name__ == '__main__':
    # Single sentence example:
    # print(("%6.2f %s" % (sentiment(text), text)))

    # No negation and booster words handled in this approach
    text = "Finn is only a tiny bit stupid and not idiotic"
    # print(("%6.2f %s" % (sentiment(text), text)))