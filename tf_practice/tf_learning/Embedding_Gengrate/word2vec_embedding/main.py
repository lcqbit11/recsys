#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import Word2Vec

if __name__ == "__main__":
    wiki = WikiCorpus('data/sw/swwiki-latest-pages-articles.xml.bz2', lemmatize=False, dictionary={})
    sentences = list(wiki.get_texts())
    params = {'size': 200, 'window': 10, 'min_count': 10,
              'workers': max(1, multiprocessing.cpu_count() - 1), 'sample': 1E-3, }
    word2vec = Word2Vec(sentences, **params)

    female_king = word2vec.most_similar_cosmul(positive='mfalme mwanamke'.split(),
                                               negative='mtu'.split(), topn=5, )
    for ii, (word, score) in enumerate(female_king):
        print("{}. {} ({:1.2f})".format(ii + 1, word, score))