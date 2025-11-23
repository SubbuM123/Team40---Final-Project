import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords

from collections import Counter
import math
import time

class InvInd():
    abstracts = {}
    titles = {}

    abstract_bv = {}
    title_bv = {}

    abstract_idf = {}
    title_idf = {}

    dataset1 = pd.read_csv("dataset1.csv")
    dataset2 = pd.read_csv("dataset2.csv")
    dataset3 = pd.read_csv("dataset3.csv")
    dataset = pd.concat([dataset1, dataset2, dataset3])
    num_rows = len(dataset)

    title_vocabulary = set()
    abstract_vocabulary = set()

    punctuations = None
    stop_words = None

    ps = nltk.stem.PorterStemmer()

    


    def __init__(self, bm25_k = 1, top_words = 500, b = 0.7):
        self.punctuations = """'",<>./?@#$%^&*_~/!()-[]{};:""" + "\\"
        self.stop_words = set(stopwords.words('english'))
        self.top_words = top_words
        self.bm25_k = bm25_k
        self.B = b

        a_list = self.dataset["abstract"].to_list()
        t_list = self.dataset["title"].to_list()
        avg_abs_length = 0
        avg_title_length = 0
        for a in a_list:
            avg_abs_length += len(a.split())
        for t in t_list:
            avg_title_length += len(t.split())

        avg_abs_length = avg_abs_length/len(a_list)
        avg_title_length = avg_title_length/len(t_list)

        self.avdl_abs = avg_abs_length
        self.avdl_title = avg_title_length


    def preprocess_data(self, title_or_abstract):
        sentences = None
        dictionary = {}
        if title_or_abstract == "title":
            sentences = self.dataset["title"].to_list()
            dictionary = self.titles
        elif title_or_abstract == "abstract":
            sentences = self.dataset["abstract"].to_list()
            dictionary = self.abstracts
        else:
            RuntimeError("Enter Title or Abstract")
        
        counter = 0
        for sentence in sentences:
            words = []
            for word in sentence.split(' '):
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
                    words.append(stem_word)
                else:
                    words.append(word)
            dictionary[counter] = words
            counter += 1

        if title_or_abstract == "title":
            self.titles = dictionary
        elif title_or_abstract == "abstract":
            self.abstracts = dictionary

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
        sentences = None
        if title_or_abstract == "title":
            vocab = self.title_vocabulary
            sentences = self.titles
        elif title_or_abstract == "abstract":
            vocab = self.abstract_vocabulary
            sentences = self.abstracts
        else:
            RuntimeError("Enter Title or Abstract")
        
        words = []
        for c in range(self.num_rows):
            for word in sentences[c]:
                words.append(word)
        
        count = Counter(words)
        sorted_count = count.most_common(self.top_words + 1)

        top_word_list = []
        for key in sorted_count:
            if key[0] == '':
                continue
            top_word_list.append(key[0])

        top_word_list = top_word_list[:self.top_words]

        doc_freq = Counter()
        for doc in sentences.values():
            unique_words = set(doc)
            doc_freq.update(unique_words)

        if title_or_abstract == "title":
            self.title_vocabulary = np.array(top_word_list)
            with open("tv.txt", "w", encoding="utf-8") as f:
                for word in top_word_list:
                    f.write(f"{word},{doc_freq[word]}\n")

        elif title_or_abstract == "abstract":
            self.abstract_vocabulary = np.array(top_word_list)
            with open("av.txt", "w", encoding="utf-8") as f:
                for word in top_word_list:
                    f.write(f"{word},{doc_freq[word]}\n")
    
    def doc_lengths(self):
        abs_lengths = []
        title_lengths = []
        a_list = self.dataset["abstract"].to_list()
        #t_list = self.dataset["title"].to_list()

        for a in a_list:
            abs_lengths.append(len(a.split()))

        # for t in a_list:
        #     abs_lengths.append(len(t.split()))

        avg_abs_length = abs_lengths.sum()/len(abs_lengths)

        self.dl = abs_lengths
        self.avdl = avg_abs_length
    
    def compute_IDF(self, title_or_abstract):
        vocab = None
        sentences = None
        if title_or_abstract == "title":
            vocab = self.title_vocabulary
            sentences = self.titles
        elif title_or_abstract == "abstract":
            vocab = self.abstract_vocabulary
            sentences = self.abstracts
        
        M = self.num_rows

        IDF = {}

        counter = 0
        for word in vocab:
            k = 1
            for sentence in sentences.values():
                if word in sentence:
                    k += 1
            
            IDF[counter] = math.log((M+1)/k)
            counter += 1
        
        if title_or_abstract == "title":
            self.title_idf = IDF
        elif title_or_abstract == "abstract":
            self.abstract_idf = IDF
    
    def text2TFIDF(self,text, title_or_abstract, q):
        vocab = None
        sentences = None
        if title_or_abstract == "title":
            vocab = self.title_vocabulary
            sentences = text
            idf = self.title_idf
            avdl = self.avdl_title
        elif title_or_abstract == "abstract":
            vocab = self.abstract_vocabulary
            sentences = text
            idf = self.abstract_idf
            avdl = self.avdl_abs

        tfidfVector = np.zeros(vocab.size)
        c = 0
        for word in vocab:
            if word in sentences:
                cwd = sentences.count(word)
                if q == False:
                    dls = len(sentences)
                    tfidfVector[c] = (((self.bm25_k + 1) * cwd)/(cwd + self.bm25_k * (1 - self.B + self.B * (dls/avdl)))) * idf[c]
                else:
                    tfidfVector[c] = (((self.bm25_k + 1) * cwd)/(cwd + self.bm25_k)) * idf[c]
            else:
                tfidfVector[c] = 0
            c += 1
        return tfidfVector
    
    def tfidf_score(self,query,doc, title_or_abstract):
        q = self.text2TFIDF(query, title_or_abstract, True)
        d = self.text2TFIDF(doc, title_or_abstract, False)

        relevance = np.dot(q, d)

        return relevance
    
    def similarity_ranking(self, query_title, query_abstract):
        query_title, query_abstract = self.preprocess_query(query_title, query_abstract)
        similarity_scores = []

        for i in range(self.num_rows):
            # if i == 0:
            #     print(self.titles[0], self.abstracts[0])
            score = self.tfidf_score(query_title, self.titles[i], "title") + self.tfidf_score(query_abstract, self.abstracts[i], "abstract")
            #score = self.tfidf_score(query_abstract, self.abstracts[i], "abstract")
            similarity_scores.append(score)
        
        return np.array(similarity_scores)
    
    def inverted_index(self):
        title_matrix = np.zeros((self.num_rows, len(self.title_vocabulary)), dtype=np.float32)
        abstract_matrix = np.zeros((self.num_rows, len(self.abstract_vocabulary)), dtype=np.float32)

        for i in range(self.num_rows):
            title_matrix[i] = self.text2TFIDF(self.titles[i], "title", False)
            abstract_matrix[i] = self.text2TFIDF(self.abstracts[i], "abstract", False)

        np.save("t.npy", title_matrix)
        np.save("a.npy", abstract_matrix)

    

if __name__ == '__main__':
    rs = InvInd(bm25_k=1.2, top_words=500)
    rs.preprocess_data("abstract")
    rs.build_vocab("abstract")
    rs.compute_IDF("abstract")
    print("Done with abstracts")
    rs.preprocess_data("title")
    rs.build_vocab("title")
    rs.compute_IDF("title")

    
    print("Done with titles")
    rs.inverted_index()

