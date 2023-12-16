# pattern-informatics-assignment

パターン情報学プログラミング課題

## 実行環境
- Windows 11
- Python 3.12.0 (@ rye 0.15.2)

## 実行方法
1. rye等で仮想環境を作成
```
$ cd pattern_informatics_assignment
$ rye sync                           # ryeが使用可能な場合
$ pip install -r requirements.lock   # その他
```
2. モジュールを実行
```
# ryeが使用可能な場合
$ rye run python -m pattern_informatics_assignment
0: Perceptron - Linear, 1: Perceptron - Quadratic, 2: Perceptron - Cubic, 3: Multinominal Logistic,
4: KNN - make_classification, 5: KNN - make_circles, 6: KNN - make_moons 7: KNN - Multiclass >>   # ターゲットを求められるので入力

# その他
$ python -m pattern_informatics_assignment 3   # 引数で直接ターゲットを指定しても可
```

# プログラムについて
- `__main__.py`：メイン処理
- `perceptron_classifier.py`：パーセプトロンを用いた識別関数の学習の実装
- `multinominal_logistic_classifier.py`：マルチクラスのロジスティック回帰（ソフトマックス回帰）モデルの実装
- `knn_classifier.py`：k近傍法の実装
- `confusion_matrix.py`：混同行列の実装
- `classifier_2d.py`：2次元データの分類器の基底クラスの定義
- `util.py`：グラフ描画関数の定義