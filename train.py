import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy
from random import shuffle
import re

class LabeledLineSentence(object):

    def __init__(self, sources):
        self.sources = sources

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    # line = re.sub(' +',' ',re.sub(r'[^A-Za-z]', ' ', line.decode('utf-8'))).lower()
                    # print(item_no, line)
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    # line = re.sub(' +',' ',re.sub(r'[^A-Za-z]', ' ', line.decode('utf-8'))).lower()
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

sources = {'cleanedHam':'ham', 'cleanedSpam':'spam'}

sentences = LabeledLineSentence(sources)

model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)

model.build_vocab(sentences.to_array())

for epoch in range(10):
    print(epoch)
    model.train(sentences.sentences_perm(),
                total_examples=model.corpus_count,
                epochs=model.iter,
    )

model.save('./spamModel.d2v')
