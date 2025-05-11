# readme

## LangChainのバージョンについて

本書はlangchain 0.1.14で執筆されました。
langchainの0.2を使う場合、以下の点に注意してください。

1. chap5のmk-rag-db-from-text.pyではHuggingFaceEmbeddingsをインポートする元をlangchain.embeddingsからlangchain_community.embeddingsに変更する。

# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

2. chap5のbm25reteiver-example.pyでは以下のワーニングが出ますが、問題なく動きます。

LangChainDeprecationWarning: The method
`BaseRetriever.get_relevant_documents` was deprecated in langchain-core
0.1.46 and will be removed in 0.3.0. Use invoke instead.
   warn_deprecated(

## その他
langchainバージョンとは関係なく、以下に注意してください。

chap5のbuild-mywikidb-bm25.pyは、ibaraki-bm25.pklを作るプログラムです。
ibaraki-bm25.pklは配布しているので動かす必要はありませんが、実際に動かすとload_datasetでエラーが出ます。
これはp.48でdatasetsを2.10.1に設定しているからです。
build-mywikidb-bm25.pyを動かすには、datasetsをdatasets-2.19.1にしてください。

$ pip install datasets==2.19.1