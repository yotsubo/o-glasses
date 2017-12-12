# o-glasses
====

（README作成中）
ニューラルネットにより256バイトのファイルの断片データでクラス分類します．

イメージは「目grep」を誰にでも体験できるツール

## Requirement

* An OS that can run Python 2.7.3 or later

#環境構築(インストール直後のUbuntu 16.04の場合)
```
sudo apt-get update
sudo apt install python-pip
pip install chainer
pip install matplotlib
pip install pillow
pip install distorm3
```

## Usage
（作成中）
学習済みモデルを使用し，ペイロードのCPUアーキテクチャの推定をする場合
```
> python o-glasses.py -im cpu -i ./payload.bin
[0, 20, 0, 2] win_gnu_gcc_32bit
```
「-im cpu」で学習済みモデル(cpu.npz, cpu.json)を読み込み

「-i ./payload.bin」でチェック対象File(./payload.bin）を読み込み

「cpu.json」はこんな感じ
```
{"num_of_types": 4, "file_types_": ["mips_32bit", "win_gnu_gcc_32bit", "arm_32bit", "powerpc_32bit"], "unit": 400}
```
「file_types_」がラベル名

「unit」が中間層のユニット数

入力ファイルから256 byteのブロックの集合を作成し，各ブロックごとにクラス分類する．
デフォルトでは，16byteごとにスライドしてブロックを作成．
ファイルサイズが512 byteの場合は，ブロックは(512-256)/16+1 = 17個作成される．

出力結果の[0, 20, 0, 2]は各ブロックのクラス分類結果を示す．

この場合の判定結果は以下のとおり．

1:「win_gnu_gcc_32bit」と判定されたブロックが20個

3:「powerpc_32bit」と判定されたブロックが2個

最終的に最も多く判定されたラベル「win_gnu_gcc_32bit」が表示される．

## Licence
Released under the MIT license  
http://opensource.org/licenses/mit-license.php

## Author

[yotsubo](https://github.com/yotsubo)
