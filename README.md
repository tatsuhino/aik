## 準備

文書のテキスト化＋分かち書き＋Doc2Vecモデルの学習は、以下の記事を参照。
http://tadaoyamaoka.hatenablog.com/entry/2017/04/29/122128
※テキスト化にはapache tikaを使用


# Word2Vecの機能を試してみる
## 似ている単語ベクトルを出力
```
# 類似単語ベクトルの取得
from gensim.models import word2vec
model   = word2vec.Word2Vec.load('wordmodel')
print(model.most_similar(positive="システム", topn=10))
print(model.most_similar(positive="プログラム", topn=10))
print(model.most_similar(positive="BASICS", topn=10))
```

## 単語ベクトルの加減算　ヒーロ＋女－男（ヒロインが出ることを期待）
```
print(model.most_similar(positive=["ヒーロー","女"],negative=["男"], topn=10))
print(model.most_similar(positive=['女', '国王'], negative=['男'], topn=10))
```

# Doc2Vecの機能を試してみる
## モデルに登録しているすべてのタグを表示
```
import gensim
from gensim import models
from gensim.models.doc2vec import TaggedDocument
model = models.Doc2Vec.load('model') #モデルの読込
print(model.docvecs.doctags) # Doc2Vecに登録されているタグの一覧を表示

print(model.docvecs.doctags)
```
## 類似文章検索をしてみる
```
print(model.docvecs.most_similar('./doc/社内/電話番号一覧.txt',topn=10,clip_start =115168))
print(model.docvecs.most_similar('./doc/青空文庫/福沢諭吉/アメリカ独立宣言.txt',topn=10,clip_start =115168))
```

## 追加学習
```
model = models.Doc2Vec.load('model')
# hoge,hugaという内容の文書をNEW_DOC1というタグで追加学習
add_doc=TaggedDocument(words=['hoge','huga'], tags=['NEW_DOC1'])

model.build_vocab([add_doc], update=True)
model.docvecs.reset_weights(model)
model.train([add_doc],total_examples=model.corpus_count, epochs=model.iter)
print(model.docvecs.doctags) #NEW_DOC1が追加されているはず
```
