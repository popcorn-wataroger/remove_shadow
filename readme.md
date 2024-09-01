# 人の顔の影をなくす生成AI（GAN）の実装
友人の影のついた顔画像とついていない画像（公開許可取得済）を学習し、
推論時には影を除去したり、付与することが可能。

# 実行環境
GoogleColab上で動作可能
使用GPUメモリ: 9～11GB
依存ライブラリ: コード要確認

# 実行方法
1. dataディレクトリに所望のデータセットをコピー．

data - shadow_dataset1 - testA（影なし）
                   　　- testB（影や光沢あり）

2. python colab_remove_shadow.py

# その他
* resultに推論結果が生成されます．
* modelsに学習モデルがあります．

