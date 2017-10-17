#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 017 Yuhei Otsubo
Released under the MIT license
http://opensource.org/licenses/mit-license.php
"""
#環境構築(インストール直後のUbuntu 16.04の場合)
# sudo apt-get update
# sudo apt install python-pip
# pip install chainer
# pip install matplotlib
# pip install pillow
# pip install distorm3
#
#基本的な使い方
#オプション無し：datasetディレクトリ(-d)のデータを使ってk-分割交差検証の実験
#-omオプション：学習済みモデルを出力
#-imオプション&-iオプション：学習済みモデルを使用し(-im)、入力ファイル(-i)の推定
#
#20170721 -iオプションでファイルの可視化ができるように改良
#20170710 目grep力を学習し、学習済みモデルを出力
#20170710 k-分割交差検証で結果を出すように改良
#20170803 BarchNormalization
#20170828 --disasm_x86で逆アセンブルしたデータセットを作成可能に

import os
import sys
import commands
import json
import random
from chainer.datasets import tuple_dataset
from chainer import Variable
from chainer import serializers
import numpy as np
from PIL import Image
from distorm3 import Decode, Decode32Bits
import binascii

try:
    import matplotlib
    import sys
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),  # n_units -> n_out
            bnorm1=L.BatchNormalization(n_units),
            bnorm2=L.BatchNormalization(n_units),
        )

    def __call__(self, x):

        #h1 = F.relu(self.l1(x))
        h1 = F.relu(self.bnorm1(self.l1(x)))
        #h1 = self.bnorm1(F.relu(self.l1(x)))

        #h2 = F.relu(self.l2(h1))
        h2 = F.relu(self.bnorm2(self.l2(h1)))
        #h2 = self.bnorm2(F.relu(self.l2(h1)))

        return self.l3(h2)




def fild_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            path = os.path.join(root, file)
            if os.path.islink(path):
            	continue
            yield path

		
def get_result(result):
	max_i=0
	for i in range(len(result)):
		if result[max_i]<result[i]:
			max_i=i
	return max_i

def bitmap_view(b):
        return b
        if b==0:
                r=0
        elif b<0x20:
                r=0x20
        elif b<0x80:
                r=0x80
        else:
                r=0xFF
        return r	

def entropy(data):
    result = []
    s = len(data)
    for x in range(256):
        n = 0
        for i in data:
            if i == x:
                n+=1
        p_i = float(n)/s
        result.append(p_i * np.log2(p_i))
    r = 0.0
    for i in result:
	if i==i:#NaNでないときの処理
            r += i
    return np.int32((-r)/8.0*255.0)

	
def main():
	parser = argparse.ArgumentParser(description='Chainer: eye-grep test')
	parser.add_argument('--batchsize', '-b', type=int, default=100,
		        help='Number of images in each mini-batch')
	parser.add_argument('--epoch', '-e', type=int, default=20,
		        help='Number of sweeps over the dataset to train')
	parser.add_argument('--k', '-k', type=int, default=3,
		        help='Number of folds (k-fold cross validation')
	parser.add_argument('--frequency', '-f', type=int, default=-1,
		        help='Frequency of taking a snapshot')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
		        help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--out', '-o', default='result',
		        help='Directory to output the result')
	parser.add_argument('--resume', '-r', default='',
		        help='Resume the training from snapshot')
	parser.add_argument('--unit', '-u', type=int, default=1000,
		        help='Number of units')
	parser.add_argument('--dataset', '-d', type=str, default="dataset",
		        help='path of dataset')
	parser.add_argument('--input', '-i', type=str, default="",
		        help='checked file name')
	parser.add_argument('--output_model', '-om', type=str, default="",
		        help='model file path')
	parser.add_argument('--input_model', '-im', type=str, default="",
		        help='model file name')
	group = parser.add_mutually_exclusive_group()
	group.add_argument('--disasm_x86', action='store_true')
	group.add_argument('--no-disasm_x86', action='store_false')
	parser.set_defaults(disasm_x86=False)

 	args = parser.parse_args()
	
	block_size = 256
        #SGD,MomentumSGD,AdaGrad,RMSprop,AdaDelta,Adam
	selected_optimizers = chainer.optimizers.Adam()

	if not args.input_model:
		#datasetディレクトリから学習モデルを作成

		path = args.dataset
		print path


		#ファイル一覧の取得

		files_file = [f for f in fild_all_files(path) if os.path.isfile(os.path.join(f))]


		#ファイルタイプのナンバリング
		file_types = {}
		file_types_ = []
		num_of_file_types = {}
		num_of_types = 0
		for f in files_file:
			#ディレクトリ名でファイルタイプ分類
			file_type = f.replace(path,"").replace(os.path.basename(f),"").split("/",1)[0]
			#print(file_type)
			if file_type in file_types:
				num_of_file_types[file_type] += 1
			else:
				file_types[file_type]=num_of_types
				file_types_.append(file_type)
				num_of_file_types[file_type] = 1
				print num_of_types,file_type
				num_of_types+=1

		#データセットの作成
		print "make dataset"
		num_of_dataset = {}
		master_dataset = []
		for f in files_file:
			ft = f.replace(path,"").replace(os.path.basename(f),"").split("/",1)[0]
			ftype = np.int32(file_types[ft])
			fin = open(f,"rb")
			bdata = fin.read()
			if args.disasm_x86:
			        l = Decode(0x4000000, bdata, Decode32Bits)
			        bdata = b''
			        for i in l:
			                #print "%-16s" % i[3]
			                #bdata+= "%-16s" % i[3]
			                b = b''
			                for c in range(16):
			                        if c < len(i[3]):
			                                b += i[3][c]
		                                else:
		                                        b += b'\0'
                                        bdata += b
                                        #print binascii.b2a_hex(b)
			fsize = len(bdata)
			if fsize < block_size:
				continue
			if ft not in num_of_dataset:
				num_of_dataset[ft] = 0

			#256バイト区切りでデータセット作成
			for c in range(0,fsize-block_size,block_size):
				offset = c*1.0/fsize
				block = bdata[c:c+block_size]
				train = np.array([np.float32(bitmap_view(ord(x))/255.0) for x in block])
				#train = np.append(train,np.float32(offset))
				train = (train,ftype)
				master_dataset.append(train)
				num_of_dataset[ft]+=1


		#データセットの情報を表示
		total_samples = 0
		total_files = 0
		total_types = 0
		print "type, num of file types, num of dataset"
		for t in num_of_dataset:
			print file_types[t],t,num_of_file_types[t],num_of_dataset[t]
			total_types+=1
			total_files+=num_of_file_types[t]
			total_samples+=num_of_dataset[t]
		print "total types", total_types
		print "total files", total_files
		print "total samples", total_samples

		print('GPU: {}'.format(args.gpu))
		print('# unit: {}'.format(args.unit))
		print('# Minibatch-size: {}'.format(args.batchsize))
		print('# epoch: {}'.format(args.epoch))
		print('')
	else:
		#学習済みモデルの入力
		f = open(args.input_model+".json","r")
		d = json.load(f)
		file_types_ = d['file_types_']
		num_of_types = d['num_of_types']
		model = L.Classifier(MLP(d['unit'], num_of_types))
		serializers.load_npz(args.input_model+".npz", model)
		if args.gpu >= 0:
			chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
			model.to_gpu()  # Copy the model to the GPU

	if args.output_model and master_dataset:
		#master_datasetが作成されていない場合、学習済みモデルは出力されない
		#学習済みモデルの作成
		# Set up a neural network to train
		# Classifier reports softmax cross entropy loss and accuracy at every
		# iteration, which will be used by the PrintReport extension below.
		model = L.Classifier(MLP(args.unit, num_of_types))
		if args.gpu >= 0:
			chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
			model.to_gpu()  # Copy the model to the GPU

		# Setup an optimizer
		optimizer = selected_optimizers
		optimizer.setup(model)

		train_iter = chainer.iterators.SerialIterator(master_dataset, args.batchsize)
		updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
		trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

		# Dump a computational graph from 'loss' variable at the first iteration
		# The "main" refers to the target link of the "main" optimizer.
		trainer.extend(extensions.dump_graph('main/loss'))

		# Write a log of evaluation statistics for each epoch
		trainer.extend(extensions.LogReport())

		# Save two plot images to the result dir
		if extensions.PlotReport.available():
			trainer.extend(
			    extensions.PlotReport(['main/loss', 'validation/main/loss'],
						  'epoch', file_name='loss.png'))
			trainer.extend(
			    extensions.PlotReport(
				['main/accuracy', 'validation/main/accuracy'],
				'epoch', file_name='accuracy.png'))

		# Print selected entries of the log to stdout
		# Here "main" refers to the target link of the "main" optimizer again, and
		# "validation" refers to the default name of the Evaluator extension.
		# Entries other than 'epoch' are reported by the Classifier link, called by
		# either the updater or the evaluator.
		trainer.extend(extensions.PrintReport(
		['epoch', 'main/loss', 'validation/main/loss',
		 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

		# Print a progress bar to stdout
		trainer.extend(extensions.ProgressBar())

		# Run the training
		trainer.run()

		#学習済みモデルの出力
		d={}
		d['file_types_'] = file_types_
		d['unit'] = args.unit
		d['num_of_types'] = num_of_types
		f = open(args.output_model+".json","w")
		json.dump(d,f)
		model.to_cpu()
		serializers.save_npz(args.output_model+".npz",model)

	elif args.input:
		if not args.input_model:
			#学習済みデータセットが指定されていない場合
			#学習済みモデルの作成
			# Set up a neural network to train
			# Classifier reports softmax cross entropy loss and accuracy at every
			# iteration, which will be used by the PrintReport extension below.
			model = L.Classifier(MLP(args.unit, num_of_types))
			if args.gpu >= 0:
				chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
				model.to_gpu()  # Copy the model to the GPU

			# Setup an optimizer
			optimizer = selected_optimizers
			optimizer.setup(model)

			train_iter = chainer.iterators.SerialIterator(master_dataset, args.batchsize)
			updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
			trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

			# Dump a computational graph from 'loss' variable at the first iteration
			# The "main" refers to the target link of the "main" optimizer.
			trainer.extend(extensions.dump_graph('main/loss'))

			# Write a log of evaluation statistics for each epoch
			trainer.extend(extensions.LogReport())

			# Save two plot images to the result dir
			if extensions.PlotReport.available():
				trainer.extend(
				    extensions.PlotReport(['main/loss', 'validation/main/loss'],
							  'epoch', file_name='loss.png'))
				trainer.extend(
				    extensions.PlotReport(
					['main/accuracy', 'validation/main/accuracy'],
					'epoch', file_name='accuracy.png'))

			# Print selected entries of the log to stdout
			# Here "main" refers to the target link of the "main" optimizer again, and
			# "validation" refers to the default name of the Evaluator extension.
			# Entries other than 'epoch' are reported by the Classifier link, called by
			# either the updater or the evaluator.
			trainer.extend(extensions.PrintReport(
			['epoch', 'main/loss', 'validation/main/loss',
			 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

			# Print a progress bar to stdout
			trainer.extend(extensions.ProgressBar())

			# Run the training
			trainer.run()

		#解析対象のデータセットの作成
		checked_dataset = []
		f=args.input
		fin = open(f,"rb")
		bdata = fin.read()
		fsize = len(bdata)
		img = Image.new('RGB', (128, (fsize+127)/128))
		for i in range(0,fsize):
			b = ord(bdata[i])
			if b == 0x00:
				c=(255,255,255)
			elif b < 0x20:
				c=(0,255,255)
			elif b<0x80:
				c=(255,0,0)
			else:
				c=(0,0,0)
			img.putpixel((i%128,i/128),c)
		img.save("bitmap.png")
		img.show()
		#256バイト区切りでデータセット作成
		img = Image.new('RGB', (128, (fsize+127)/128))
		l=16
		for c in range(0,fsize-block_size,l):
			offset = c*1.0/fsize
			block = bdata[c:c+block_size]
			block_ = [ord(x) for x in block]
			e = entropy(block_)
                        for j in range(0,l):
        			img.putpixel(((c+j)%128,(c+j)/128),(e,e,e))
        		if args.disasm_x86:
			        m = Decode(0x4000000, block, Decode32Bits)
			        block = b''
			        for i in m:
			                b = b''
			                for c in range(16):
			                        if c < len(i[3]):
			                                b += i[3][c]
		                                else:
		                                        b += b'\0'
                                        block += b
                                block = block[:block_size]
			
			train = np.array([np.float32(bitmap_view(ord(x))/255.0) for x in block])
			#train = np.append(train,np.float32(offset))
			checked_dataset.append(train)
		img.save("entropy.png")
		img.show()
		#解析対象のファイルの分類結果を表示
		img = Image.new('RGB', (128, (fsize+127)/128))
		col = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
		print args.input
		results = [0 for i in range(num_of_types)]
		for i in range(len(checked_dataset)):
			#predictor = F.softmax(model.predictor(np.array([checked_dataset[i]]).astype(np.float32))).data[0]
#			print predictor
#			result = get_result(predictor)
                        with chainer.using_config('train', False):
        			result = model.predictor(np.array([checked_dataset[i]]).astype(np.float32)).data.argmax(axis=1)[0]
			results[result]+=1
                        for j in range(0,l):
        			img.putpixel(((i*l+j)%128,(i*l+j)/128),col[result])
		print results,file_types_[get_result(results)]
		img.save("v.png")
		img.show()
	else:
		#k-分割交差検証
		random.shuffle(master_dataset)
		k=args.k
		for i in range(k):
			train_dataset = []
			test_dataset = []
			flag = True
			c = 0
			for train in master_dataset:
				if c<total_samples/k*i:
					train_dataset.append(train)
				elif c>=total_samples/k*(i+1):
					train_dataset.append(train)
				else:
					test_dataset.append(train)
				c+=1


			# Set up a neural network to train
			# Classifier reports softmax cross entropy loss and accuracy at every
			# iteration, which will be used by the PrintReport extension below.
			model = L.Classifier(MLP(args.unit, num_of_types))
			if args.gpu >= 0:
				chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
				model.to_gpu()  # Copy the model to the GPU

			# Setup an optimizer
			optimizer = selected_optimizers
			optimizer.setup(model)

			# Load the dataset
			train = train_dataset
			test = test_dataset


			train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
			test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
						                 repeat=False, shuffle=False)

			# Set up a trainer
			updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
			trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

			# Evaluate the model with the test dataset for each epoch
			trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

			# Dump a computational graph from 'loss' variable at the first iteration
			# The "main" refers to the target link of the "main" optimizer.
			trainer.extend(extensions.dump_graph('main/loss'))

			# Take a snapshot for each specified epoch
			frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
			trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

			# Write a log of evaluation statistics for each epoch
			trainer.extend(extensions.LogReport())

			# Save two plot images to the result dir
			if extensions.PlotReport.available():
				trainer.extend(
				    extensions.PlotReport(['main/loss', 'validation/main/loss'],
							  'epoch', file_name="{0:02d}".format(i)+'loss.png'))
				trainer.extend(
				    extensions.PlotReport(
					['main/accuracy', 'validation/main/accuracy'],
					'epoch', file_name="{0:02d}".format(i)+'accuracy.png'))

			# Print selected entries of the log to stdout
			# Here "main" refers to the target link of the "main" optimizer again, and
			# "validation" refers to the default name of the Evaluator extension.
			# Entries other than 'epoch' are reported by the Classifier link, called by
			# either the updater or the evaluator.
			trainer.extend(extensions.PrintReport(
			['epoch', 'main/loss', 'validation/main/loss',
			 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

			# Print a progress bar to stdout
			trainer.extend(extensions.ProgressBar())

			if args.resume:
				# Resume from a snapshot
				chainer.serializers.load_npz(args.resume, trainer)


			# Run the training
			trainer.run()




if __name__ == '__main__':
    main()
