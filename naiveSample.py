# coding: UTF-8

import numpy as np
from sklearn.naive_bayes import GaussianNB # ガウシアン
X = np.array([[1,2,3,4,5,6,7,8],
              [1,1,3,4,5,6,6,7],
              [2,1,2,4,5,8,8,8]]) # 特徴ベクトル
y = np.array([1, 2, 3]) # そのラベル
t = np.array([2,2,4,5,6,8,8,8]) # テストデータ

clf = GaussianNB() # 正規分布を仮定したベイズ分類
clf.fit(X, y) # 学習をする
print(clf.predict(t)) # => [3]
