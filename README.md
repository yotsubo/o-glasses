# o-glasses (MLP)


o-glasses is an intuitive binary classification and visualization tool with machine learning.

The concept of o-glasses is easy-to-do eye-grep.

ニューラルネットにより256バイトのファイルの断片データでクラス分類します．

イメージは「目grep」を誰にでも体験できるツール

## Requirement

* An OS that can run Python 2.7.3 or later
* [Chainer](https://github.com/chainer/chainer) 2.0 or later
* matplotlib
* pillow
* distorm3

## Installation
Ubuntu 16.04
```
>sudo apt-get update
>sudo apt install python-pip
>pip install chainer
>pip install matplotlib
>pip install pillow
>pip install distorm3
```

## Usage
####  Mode1(data set validation:データセットの検証)
The following command shows an example of validating data set you prepare.
o-glasses divides each file into 256 byte and creates 256-dimensional feature vector from 256 Byte raw data normalization and do validation using k-fold cross validation.  
```
>python o-glasses.py -d path-to-data-set 
```
You can use the following option when running.
* The `-u` option set designated value as intermediate layer unit.
* The `-e` option specify number of epoch[20].
* The `-k` option specify k-fold cross validation[3].

#### Mode2(building trained model:学習済モデルの生成)
In this case, O-glasses build a trained model.
When it finished, (model name).json and (model_name).npz will be created.
```
>python o-glasses.py -om model_name -i path-to-data-set
```
* The `-om` spcify output name of trained model.
#### Mode3(file estimation:ファイル推定)
The following is the example of cpu architecture estimation in accordance with trained model you specify.

学習済みモデルを使用し，ペイロードのCPUアーキテクチャの推定をする場合

```
> python o-glasses.py -im cpu -i ./payload.bin
[0, 20, 0, 2] win_gnu_gcc_32bit
```
* The `-im` option specify trained model(cpu.npz and cpu.json).
* The `-i` option specify a file for estimation check.


* 「-im cpu」で学習済みモデル(cpu.npz, cpu.json)を読み込み
* 「-i ./payload.bin」でチェック対象File(./payload.bin）を読み込み

cpu.json goes like this(「cpu.json」はこんな感じ):
```
{"num_of_types": 4, "file_types_": ["mips_32bit", "win_gnu_gcc_32bit", "arm_32bit", "powerpc_32bit"], "unit": 400}
```
'file_types_':Label(「file_types_」がラベル名)

'unit': the number of neuron for hidden layer(「unit」が中間層のユニット数)


O-glasses split input file into 256-size data blob and classify every blob.
In default, it does every 16 byte slide.
If file size is 512-byte, the number of block is (512-256)/16+1 = 17.

入力ファイルから256 byteのブロックの集合を作成し，各ブロックごとにクラス分類する．
デフォルトでは，16byteごとにスライドしてブロックを作成．
ファイルサイズが512 byteの場合は，ブロックは(512-256)/16+1 = 17個作成される．

[0, 20, 0, 2] shown in output the result of classification for every blob.

出力結果の[0, 20, 0, 2]は各ブロックのクラス分類結果を示す．

In this case(この場合の判定結果は以下のとおり．), 

* No blobs are estimated as 'mips_32bit'.
* 20 blobs are estimated as 'win_gnu_gcc_32bit'.
* No blobs are estimated as 'arm_32bit'.
* 2  blobs are estimated as 'powerpc_32bit'


* 1:「win_gnu_gcc_32bit」と判定されたブロックが20個
* 3:「powerpc_32bit」と判定されたブロックが2個

The most estimated label 'win_gnu_gcc_32bit' is displayed as a result of file estimation.

最終的に最も多く判定されたラベル「win_gnu_gcc_32bit」が表示される．

## Licence
Released under the MIT license  
http://opensource.org/licenses/mit-license.php

## Author

[yotsubo](https://github.com/yotsubo)
