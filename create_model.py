import gensim
import smart_open
import argparse
import codecs

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('--save_model', '-s', default='model', type=str)
args = parser.parse_args()

def read_corpus(fname):
  with codecs.open(fname, encoding="utf-8", errors='ignore') as f:
    for i, line in enumerate(f):
      last = line.split(" ")[-1].rstrip()
      tagname = last+"_"+str(i)
      print(tagname)
      yield gensim.models.doc2vec.TaggedDocument(line.rstrip().split(" "), [tagname])


train_corpus = list(read_corpus(args.input))

model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=5, epochs=55 )

model.build_vocab(train_corpus)
print("train complete")

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)

model.save(args.save_model)