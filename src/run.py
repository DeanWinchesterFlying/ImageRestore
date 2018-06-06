from dataset import DataSet
from model.ducnn import DuCNN
from model.ircnn import IRCNN
from model.resicnn import ResiCNN
from skimage import io, transform
import numpy as np
import os


def square_loss(a, b):
	loss = np.sum(np.sqrt(np.sum(np.sum(np.square(a - b), axis=1), axis=0)))
	return loss

train = True

model_index = 1
channels = 1

for i in range(3):
	model = ResiCNN(channels=channels)
	model.bulid()
	#if i > 0:
	model.load('./model/config%d' % model_index, './model/model%d' % model_index)
	model.compile()
	if train:
		data = DataSet('/home/data-crawler/image/train1/')
		print('training...')
		for t in data.get_data(channels=channels):
			model.train(t, epochs=1)
		model.save('./model/config%d' % model_index, './model/model%d' % model_index)

	data = DataSet('/home/data-crawler/image/train2/')
	origin = 0.0
	restore = 0.0
	cnt = 0
	xs = []
	ys = []

	for t in data.get_data(channels=channels):
		if cnt > 500:
			break
		for x, y in zip(t[0], t[1]):
			cnt += 1
			xs.append(x)
			ys.append(y)
			y_ = model.predict(x)[0]
			y_[y_ > 1.0] = 1.0
			y_[y_ < 0.0] = 0.0
			origin += square_loss(y, x)
			restore += square_loss(y, y_)
			if cnt == 1:
				io.imsave('result/ori%d.png' % cnt, y.reshape([y.shape[0], y.shape[1]]))
				io.imsave('result/corr%d.png' % cnt, x.reshape([y.shape[0], y.shape[1]]))
				io.imsave('result/res%d.png' % cnt, y_.reshape([y.shape[0], y.shape[1]]))

	xs = np.array(xs)
	ys = np.array(ys)
	print('avg origin loss: %f, avg restore loss: %f' % (origin / cnt, restore / cnt))
	print('after epoch %d, eval loss is %f' % (i, model.model.evaluate(x=xs, y=ys, batch_size=4)))