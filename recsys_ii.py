import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords

from collections import Counter
import math
import time

class RecSys_II():
    abstracts = {}
    titles = {}

    abstract_bv = {}
    title_bv = {}

    abstract_idf = []
    title_idf = []

    dataset1 = pd.read_csv("data/dataset1.csv")
    dataset2 = pd.read_csv("data/dataset2.csv")
    dataset3 = pd.read_csv("data/dataset3.csv")
    dataset = pd.concat([dataset1, dataset2, dataset3])
    num_rows = len(dataset)

    abstract_vocabulary = {}
    title_vocabulary = {}

    punctuations = None
    stop_words = None

    ps = nltk.stem.PorterStemmer()


    def __init__(self, bm25_k = 1, top_words = 500, b = 0.7):
        self.punctuations = """'",<>./?@#$%^&*_~/!()-[]{};:""" + "\\"
        self.stop_words = set(stopwords.words('english'))
        self.top_words = top_words
        self.bm25_k = bm25_k
        self.B = b
        
        self.title_idf = np.load("t.npy").astype(np.float32)[:, :top_words]
        self.abstract_idf = np.load("a.npy").astype(np.float32)[:, :top_words]

    def preprocess_query(self, query_title, query_abstract):
        counter = 0
        q_title = []
        for word in query_title.split(' '):
            word = word.lower()
            word = word.strip()
            if word == " " or word == '':
                continue
            for i in range(len(self.punctuations)):
                if self.punctuations[i] in word:
                    word = word.replace(self.punctuations[i], '')
            if word in self.stop_words:
                continue
            stem_word = self.ps.stem(word)
            if stem_word:
                q_title.append(stem_word)
            else:
                q_title.append(word)
        
        q_abstract = []
        for word in query_abstract.split(' '):
            word = word.lower()
            word = word.strip()
            if word == " " or word == '':
                continue
            for i in range(len(self.punctuations)):
                if self.punctuations[i] in word:
                    word = word.replace(self.punctuations[i], '')
            if word in self.stop_words:
                continue
            stem_word = self.ps.stem(word)
            if stem_word:
                q_abstract.append(stem_word)
            else:
                q_abstract.append(word)
        
        return q_title, q_abstract

    def build_vocab(self, title_or_abstract):
        with open("vocabs/tv.txt", "r", encoding="utf-8") as f:
            l = 0
            for line in f:
                if l >= self.top_words:
                    break
                line = line.strip()
                word, count = line.split(",")
                self.title_vocabulary[word.strip()] = int(count.strip())
                l += 1

        with open("vocabs/av.txt", "r", encoding="utf-8") as f:
            l = 0
            for line in f:
                if l >= self.top_words:
                    break
                line = line.strip()
                word, count = line.split(",")
                self.abstract_vocabulary[word.strip()] = int(count.strip())
                l += 1
    
    def text2TFIDF(self,text, title_or_abstract, q):
        vocab = None
        sentences = None
        if title_or_abstract == "title":
            vocab = self.title_vocabulary
            sentences = text
            idf = self.title_idf
        elif title_or_abstract == "abstract":
            vocab = self.abstract_vocabulary
            sentences = text
            idf = self.abstract_idf
        M = self.num_rows
        tfidfVector = np.zeros(len(vocab))
        c = 0
        for word in vocab:
            if word in sentences:
                cwd = sentences.count(word)
                tfidfVector[c] = (((self.bm25_k + 1) * cwd)/(cwd + self.bm25_k)) * math.log((M+1)/vocab[word])
            else:
                tfidfVector[c] = 0
            c += 1
        return tfidfVector[:self.top_words]
    
    def tfidf_score(self,query_vec,doc_vec, title_or_abstract):
        relevance = np.dot(query_vec, doc_vec)
        return relevance
    
    def similarity_ranking(self, query_title, query_abstract):
        query_title, query_abstract = self.preprocess_query(query_title, query_abstract)
        q_title = self.text2TFIDF(query_title, "title", False)
        q_abstract = self.text2TFIDF(query_title, "abstract", False)

        similarity_scores = []
        for i in range(self.num_rows):
            score = 0.66*self.tfidf_score(q_title, self.title_idf[i], "title") + 0.33*self.tfidf_score(q_abstract, self.abstract_idf[i], "abstract")
            similarity_scores.append(score)
        
        return np.array(similarity_scores)
    
    

if __name__ == '__main__':
    k = 1.2
    rs = RecSys_II(bm25_k=k, top_words=500)

    rs.build_vocab("abstract")
    rs.build_vocab("title")

    qts = ["Implicit semantic text retrieval and distributed implementation for rural medical care"
    ,"A systematic study of double auction mechanisms in cloud computing"
    ,"Memory Efficiency of Parallel Programs and Memory Bounded Speedup"
    ,"Computer-Aided Design of Machine Learning Algorithm: Training Fixed-Point Classifier for On-Chip Low-Power Implementation"
    ,"Web Service Clustering Using Relational Database Approach"]

    mrr = 0
    for qt in qts:
        ss = rs.similarity_ranking(qt, qt)

        top100_indices = np.argsort(ss)[-100:][::-1]   
        top100_scores = ss[top100_indices]

        for rank, idx in enumerate(top100_indices, start=1):
            if rs.dataset["title"].iloc[idx] == qt:
                mrr += 1 / rank
                break
        
    with open("hp.txt", "a") as file:
        file.write("k = " + str(k) + "\n")
        file.write("MRR = " + str(mrr/5) + "\n")
        file.write("--------------------\n")
