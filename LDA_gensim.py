#Read the datafile

import pandas as pd
data = pd.read_csv('github_comments.csv')
comments = data['comment']

#Text Data Pre-processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()
import string
import re

stop_words=stopwords.words("english")
stop_words.extend(['http','thank','think', 'good','fix','merg','nice','essenti','tinker','catch','rebas','master','wdyt','would','done' ,'pleas' ,'need','could','know','look','also','close','github','comment','maybe','like','definit','good','make','commit'])
def clean(doc):
    names = re.sub(r'@[a-zA-Z0-9_\-\.]+', 'na', doc)
    tokenized=nltk.word_tokenize(names) # step 1: tokenize
    lowercase=[i.lower() for i in tokenized] # step 2: convert to lower case
    punc_free=[i for i in lowercase if not i in string.punctuation] # step 4: get rid of the punctuations
    normalized= [stemmer.stem(i) for i in punc_free]
    stop_free=[i for i in normalized if i not in stop_words and len(i) > 3] # step 3: get rid of stop words# step 5: stem the each words
    return stop_free
    
 #Passing our raw data to function
corpus_clean= [clean(i) for i in comments] # step 6: format the corpus as a list of a list
#Running model for 1000 iterations, num_topics=10, num_words=4

from gensim import corpora
dictionary = corpora.Dictionary(corpus_clean) # this is all words. All the words appearing in all the documents



# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above
doc_term_matrix = [dictionary.doc2bow(doc) for doc in corpus_clean] # conver to a bag of words.


# Creating the object for LDA model using gensim library
import gensim
Lda = gensim.models.ldamodel.LdaModel

# Running and Training LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics= 10, id2word = dictionary, passes= 1000)

# print most significant topics, and the most important words
print(ldamodel.print_topics(num_topics=10, num_words=4 ))

Output:
[(0, '0.015*"argument" + 0.010*"period" + 0.008*"favor" + 0.006*"repeat"'), 
(1, '0.018*"test" + 0.015*"work" + 0.010*"case" + 0.010*"onli"'), 
(2, '0.018*"class" + 0.014*"method" + 0.012*"chang" + 0.011*"valu"'), 
(3, '0.046*"test" + 0.014*"right" + 0.009*"feel" + 0.009*"reason"'), 
(4, '0.109*"code" + 0.107*"chang" + 0.106*"issu" + 0.104*"want"'), 
(5, '0.028*"releas" + 0.023*"analyz" + 0.012*"modul" + 0.009*"bridg"'), 
(6, '0.032*"remov" + 0.024*"line" + 0.015*"appli" + 0.012*"miss"'), 
(7, '0.018*"sign" + 0.017*"veri" + 0.013*"licens" + 0.012*"resolv"'), (
8, '0.028*"spring" + 0.021*"session" + 0.020*"updat" + 0.016*"chang"'), 
(9, '0.537*"reloc" + 0.026*"jenkin" + 0.022*"retest" + 0.007*"metadata"')]
