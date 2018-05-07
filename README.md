## 準備
import gensim
import time
from gensim import models
model = models.Doc2Vec.load('model')
#from gensim.models.doc2vec import TaggedDocument

print(model.docvecs.doctags)



start = time.time()
print(model.docvecs.most_similar('./aozora/tika/XXXX.xls_tika.txt',topn=10))
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")


for x in dir(model):
  print(x)


##Word2Vecの機能を試してみる
# 日本に似ている単語を出してみる
print(model.most_similar(positive="システム", topn=10))
print(model.most_similar(positive="プログラム", topn=10))
print(model.most_similar(positive="BASICS", topn=10))

# ヒーロ＋女－男（ヒロインが出ることを期待）
print(model.most_similar(positive=["ヒーロー","女"],negative=["男"], topn=10))

print(model.most_similar(positive=['女', '国王'], negative=['男'], topn=10))


##Doc2Vecの機能を試してみる
# モデルに登録しているすべてのタグを表示
print(model.docvecs.doctags)
# 類似文章検索をしてみる
print(model.docvecs.most_similar('./aozora/社内doc/電話番号一覧.txt',topn=10,clip_start =115168))
print(model.docvecs.most_similar('./aozora/青空文庫/福沢諭吉/アメリカ独立宣言.txt',topn=10,clip_start =115168))

##Doc2Vecのオンライン学習

add_doc=TaggedDocument(words=['mogemoge','hugahuga'], tags=['NEW_DOC1'])
model.build_vocab([add_doc], update=True)
model.docvecs.reset_weights(model)
model.train([add_doc],total_examples=model.corpus_count, epochs=model.iter)
print(model.docvecs.doctags)#NEW_DOC1が追加されているはず
print(model.docvecs.most_similar('SENT1',topn=10,clip_start =115168))



＜登録のない単語＞
from gensim import models
model = models.Doc2Vec.load('model')

print(model.infer_vector("hogehoge"))
print(model.docvecs.most_similar([model.infer_vector("hogehoge")],topn=1000))




model.train(sentences, updated_count, epochs=model.iter)

＜登録のない単語の追加＞
add_doc=TaggedDocument(words=['mogemoge','hugahuga'], tags=['SENT1'])
add_doc=TaggedDocument(words=['hino','tatsuya'], tags=['SENT2'])
add_doc=TaggedDocument(words=['sinagawa','tatsuya'], tags=['SENT3'])
model.build_vocab([add_doc], update=True)
model.docvecs.reset_weights(model)
model.train([add_doc],total_examples=model.corpus_count, epochs=model.iter)