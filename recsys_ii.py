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

    dataset1 = pd.read_csv("dataset1.csv")
    dataset2 = pd.read_csv("dataset2.csv")
    dataset3 = pd.read_csv("dataset3.csv")
    dataset = pd.concat([dataset1, dataset2, dataset3])
    num_rows = len(dataset)

    abstract_vocabulary = {}
    title_vocabulary = {}

    punctuations = None
    stop_words = None

    ps = nltk.stem.PorterStemmer()


    def __init__(self, bm25_k = 10, top_words = 500):
        self.punctuations = """'",<>./?@#$%^&*_~/!()-[]{};:""" + "\\"
        self.stop_words = set(stopwords.words('english'))
        self.top_words = top_words
        self.bm25_k = bm25_k
        self.B = 0.5

        with open("t.txt", 'r') as file:
            vec = file.readline()
            while vec:
                tfidf = np.zeros(top_words)
                vec = vec.split(",")
                for v in range(top_words):
                    
                        tfidf[v] = float(vec[v])
                    
                self.title_idf.append(tfidf)
                vec = file.readline()
        with open("a.txt", 'r') as file:
            vec = file.readline()
            while vec:
                tfidf = np.zeros(top_words)
                vec = vec.split(",")
                for v in range(top_words):
                    tfidf[v] = float(vec[v])
                self.abstract_idf.append(tfidf)
                vec = file.readline()

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
            q_abstract.append(word)
        
        return q_title, q_abstract

    def build_vocab(self, title_or_abstract):
        with open("tv.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                word, count = line.split(",")
                self.title_vocabulary[word.strip()] = int(count.strip())

        with open("av.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                word, count = line.split(",")
                self.abstract_vocabulary[word.strip()] = int(count.strip())
    
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
        return tfidfVector
    
    def tfidf_score(self,query,doc_vec, title_or_abstract):
        q = self.text2TFIDF(query, title_or_abstract, False)
        d = doc_vec

        relevance = np.dot(q, d)

        return relevance
    
    def similarity_ranking(self, query_title, query_abstract):
        query_title, query_abstract = self.preprocess_query(query_title, query_abstract)
        similarity_scores = []

        for i in range(self.num_rows):
            score = self.tfidf_score(query_title, self.title_idf[i], "title") + self.tfidf_score(query_abstract, self.abstract_idf[i], "abstract")
            similarity_scores.append(score)
        
        return np.array(similarity_scores)
    
    

if __name__ == '__main__':
    t = time.time()
    rs = RecSys_II(bm25_k=1.2, top_words=15)
    rs.build_vocab("abstract")
    print("Built abstract")
    rs.build_vocab("title")
    print("Built title")
    #qt and qa are user inputs
    qt = "Entity resolution with iterative blocking"

    qa = "Entity Resolution (ER) is the problem of identifying which records in a database refer to the same real-world entity. " \
    "An exhaustive ER process involves computing the similarities between pairs of records, which can be very expensive for large " \
    "datasets. Various blocking techniques can be used to enhance the performance of ER by dividing the records into blocks in multiple " \
    "ways and only comparing records within the same block. However, most blocking techniques process blocks separately and do not " \
    "exploit the results of other blocks. In this paper, we propose an  iterative blocking framework  where the ER results of blocks " \
    "are reflected to subsequently processed blocks. Blocks are now iteratively processed until no block contains any more matching " \
    "records. Compared to simple blocking, iterative blocking may achieve higher accuracy because reflecting the ER results of blocks " \
    "to other blocks may generate additional record matches. Iterative blocking may also be more efficient because processing a block now " \
    "saves the processing time for other blocks. We implement a scalable iterative " \
    "blocking system and demonstrate that iterative blocking can be more accurate and efficient than blocking for large datasets."

    ss = rs.similarity_ranking(qt, qa)
    print("Built ss")
    print(ss.mean(), ss.var(), np.median(ss), np.max(ss), np.min(ss))
    # top5 = np.sort(ss)[-10:]
    # print(top5)
    top10_indices = np.argsort(ss)[-10:][::-1]   # reverse for descending order

    # Get their corresponding similarity scores
    top10_scores = ss[top10_indices]

    print("Top 10 similarity scores and indices:")
    for idx, score in zip(top10_indices, top10_scores):
        print(f"Doc #{idx}: score = {score}")
        print(rs.dataset["title"].iloc[idx])
    t2 = time.time()
    print(t2 - t)
    #print(rs.abstract_vocabulary)
