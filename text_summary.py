import nltk
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords
import math

stopwords = set(stopwords.words("english"))
ps = PorterStemmer()

# calculating word freq
def word_freq(text):
    words = word_tokenize(text)
    freq = {}

    for i in words:
        i = ps.stem(i.lower())
        if (i not in stopwords) and (len(i) > 1):
            if i not in freq.keys():
                freq[i] = 1
            else:
                freq[i] += 1
    
    return freq


# calculating word freq for each sentence
def sent_word_freq(text):
    freq = {}

    for sent in sent_tokenize(text):
        freq[sent] = word_freq(sent)
    
    return freq


# tf    =   no of occurance of term in doc / total no of term in doc.


def tf(freq):
    tf_matrix = {}

    for sent, w_frq in freq.items():
        temp = {}
        l = len(w_frq)

        for word in w_frq.keys():
            temp[word] = w_frq[word] / l
        tf_matrix[sent] = temp
    
    return tf_matrix


# no of doc with term in them
def doc_frequency(freq):
    doc_table = {}
    
    for sent, w_freq in freq.items():
    
        for word in w_freq.keys():
            if word not in doc_table.keys():
                doc_table[word] = 1
            else:
                doc_table[word] += 1
    
    return doc_table


# idf =  loge(total no of doc in corpus/no of doc with term in them)


def idf(freq, doc):
    l = len(freq.keys())
    idf_matrix = {}

    for sent, w_freq in freq.items():
        temp = {}
    
        for word in w_freq.keys():
            temp[word] = math.log(l / doc[word])
        idf_matrix[sent] = temp
    
    return idf_matrix


# tf_idf = tf * idf


def tf_idf(tf_matrix, idf_matrix):
    weight = {}
    
    for (sent1, w_score1), (sent2, w_score2) in zip(tf_matrix.items(), idf_matrix.items()):
        temp = {}
        
        for (word1, score1), (word2, score2) in zip(w_score1.items(), w_score2.items()):
            temp[word1] = score1 * score2
        weight[sent1] = temp
    
    return weight


# assigning scores to sentences


def sentence_score(tf_idf_matrix):
    sent_score = {}
    
    for sent, w_score in tf_idf_matrix.items():
    
        for word, score in w_score.items():
            if sent not in sent_score.keys():
                sent_score[sent] = score
            else:
                sent_score[sent] += score
    
    return sent_score


# getting average score


def average_score(sent_score):
    sum = 0
    l = len(sent_score)
    
    for score in sent_score.values():
        sum += score
    avrg = sum / l
    
    return avrg


# getting summary based on threshold


def get_summary(sentences, sentence_score, threshold):
    summary = ""
    
    for sent in sentences:
        if (sent in sentence_score) and sentence_score[sent] >= threshold:
            summary += " " + sent
    
    return summary
