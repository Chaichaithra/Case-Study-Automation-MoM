import gensim
from gensim import corpora

#output_filepath = 'C:/Users/chait/Desktop/Casestudy/minutesOM/'

with open('C:/Users/chait/Desktop/Casestudy/minutesOM/SRH Hochschule Heidelberg 1Text.txt') as f:
    lines = f.readlines()
print(lines)
doc = ' '.join(lines)
print(doc)

doc_complete = [doc]


from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete] 

import gensim
from gensim import corpora
def topics():
    dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)
    return(ldamodel.print_topics(num_topics=3, num_words=3))


# Creating the term dictionary of our courpus, where every unique term is assigned an index. 


if __name__=='__main__':
    topics()