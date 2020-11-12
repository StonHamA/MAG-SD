import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as t
import os
import copy
import argparse
import numpy as np
import pandas as pd
import random
import cv2
import albumentations
import csv

from dataset.Dataloader import dataset_loader
from dataset.covid19 import XRayCenterCrop, XRayResizer, CLAHE, ZscoreNormalize, triDim
from core.base import base
from core.utils import make_dirs, Logger, os_walk, time_now, analyze_names_and_meter, analyze_meter_4_csv, batch_augment
from core.train_new import train_a_ep, test_a_ep #test
from core.localize import localize,localize_penumonia

def count_instance_num(dataset):
	ids = [str(i) for i in dataset.indices]
	labels = pd.DataFrame(dataset.dataset.labels)
	labels = labels.iloc[ids].values
	instance_nums = sum(labels)
	return instance_nums

def main(config):

	# environments
	make_dirs(config.save_path)
	make_dirs(os.path.join(config.save_path, 'logs/'))
	make_dirs(os.path.join(config.save_path, 'model/'))
	make_dirs(os.path.join(config.save_path, 'dataset/'))
	os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'


	# loaders
	transform = torchvision.transforms.Compose([XRayResizer(config.image_size),
												CLAHE(clip_limit=4.0, tile_grid_size=(4, 4)),
												t.ToPILImage(),
												])

	aug = torchvision.transforms.RandomApply([t.ColorJitter(brightness=0.5, contrast=0.7),
                                              t.RandomRotation(120),
                                              t.RandomResizedCrop(config.image_size, scale=(0.6, 1.0), ratio=(0.75, 1.33), interpolation=2),
                                              t.RandomHorizontalFlip(),
                                              t.RandomVerticalFlip(),
                                          ], p=0.5)
	aug = t.Compose([aug,
					 ZscoreNormalize(),
					 t.ToTensor(),
					 ])

	loader= dataset_loader(config, transform, aug)
	# base
	Base = base(config, loader)
	# logger
	logger = Logger(os.path.join(os.path.join(config.save_path, 'logs/'), 'logging.txt'))
	logger(config)

	# automatically resume model from the latest one
	start_epoch = 0
	pathologies = loader.train_set.dataset.pathologies
	count_train = count_instance_num(loader.train_set)
	count_val = count_instance_num(loader.val_set)
	logger(('all_train', len(loader.train_set)+len(loader.val_set), 'class:', pathologies, '  num:', count_train + count_val))
	logger(('train:', len(loader.train_set), 'class:', pathologies, '  num:', count_train))
	logger(('validation:', len(loader.val_set), 'class:', pathologies, '  num:', count_val))
	logger(pathologies)




	root, _, files = os_walk(Base.save_model_path)
	if len(files) > 0:
		# get indexes of saved models
		indexes = []
		for file in files:
			indexes.append(int(file.replace('.pkl', '').split('_')[-1]))

		# remove the bad-case and get available indexes
		model_num = len(Base.model_list)
		available_indexes = copy.deepcopy(indexes)
		for element in indexes:
			if indexes.count(element) < model_num:
				available_indexes.remove(element)

		available_indexes = sorted(list(set(available_indexes)), reverse=True)
		unavailable_indexes = list(set(indexes).difference(set(available_indexes)))

		if len(available_indexes) > 0 and config.mode != '5fold':  # resume model from the latest model
			Base.resume_model(available_indexes[0])
			start_epoch = available_indexes[0]
			logger('Time: {}, automatically resume training from the latest step (model {})'.
				   format(time_now(), available_indexes[0]))
			logger('Time: {},read train indices from /dataset'.format(time_now()))
			logger('Time: {},read train indices from /dataset'.format(time_now()))
			loader.train_set.indices = np.load(os.path.join(config.save_path, 'dataset', 'train.npy'))
			loader.train_set.dataset.idxs = np.load(os.path.join(config.save_path, 'dataset', 'train_idx.npy'))
			loader.train_set.dataset.labels = np.load(os.path.join(config.save_path, 'dataset', 'train_labels.npy'))

			loader.val_set.indices = np.load(os.path.join(config.save_path, 'dataset', 'test.npy'), )
			loader.val_set.dataset.idxs = np.load(os.path.join(config.save_path, 'dataset', 'test_idx.npy'))
			loader.val_set.dataset.labels = np.load(os.path.join(config.save_path, 'dataset', 'test_labels.npy'))


			count_train = count_instance_num(loader.train_set)
			count_val = count_instance_num(loader.val_set)
			logger(('all: num:',count_train + count_val))
			logger(('train: num:', count_train))
			logger(('test: num:', count_val))
	else:
		logger('Time: {}, there are no available models'.format(time_now()))
		logger('Time: {},write train indices in /dataset/train.npy'.format(time_now()))
		logger('Time: {},write train indices in /dataset/train_idx.npy'.format(time_now()))
		logger('Time: {},write train indices in /dataset/train_labels.npy'.format(time_now()))
		logger('Time: {},write test indices in /dataset/test.npy'.format(time_now()))
		logger('Time: {},write test indices in /dataset/test_idx.npy'.format(time_now()))
		logger('Time: {},write test indices in /dataset/test_labels.npy'.format(time_now()))

		np.save(os.path.join(config.save_path, 'dataset', 'train.npy'), np.array(loader.train_set.indices))
		np.save(os.path.join(config.save_path, 'dataset', 'train_idx.npy'), np.array(loader.train_set.dataset.idxs))
		np.save(os.path.join(config.save_path, 'dataset', 'train_labels.npy'), loader.train_set.dataset.labels)
		np.save(os.path.join(config.save_path, 'dataset', 'test.npy'), np.array(loader.val_set.indices))
		np.save(os.path.join(config.save_path, 'dataset', 'test_idx.npy'), np.array(loader.val_set.dataset.idxs))
		np.save(os.path.join(config.save_path, 'dataset', 'test_labels.npy'), loader.val_set.dataset.labels)

	if config.mode == 'train':
		# get all the id in dataset
		dataset_to_split = [i for i, _ in enumerate(loader.all_set)]
		# random split them to 5 folds
		train_ids_by_fold = []
		test_ids_by_fold = []
		test_cache = []
		for data_id in range(5):
			train_cache = list(set(dataset_to_split) - set(test_cache))
			test_part = random.sample(train_cache, int(len(dataset_to_split) / 5))
			test_cache = test_cache + test_part
			train_part = list(set(dataset_to_split) - set(test_part))
			train_ids_by_fold.append(train_part)
			test_ids_by_fold.append(test_part)

		for fold_id in range(5):
			# re-initialize after final test
			start_epoch = 0
			Base = base(config, loader)
			loader.train_set.indices = train_ids_by_fold[fold_id]
			loader.val_set.indices = test_ids_by_fold[fold_id]

			logger('**********' * 3 + '5fold_train_fold_' + str(fold_id) + '**********' * 3)
			for current_step in range(start_epoch, config.joint_training_steps):
				# save model every step. extra models will be automatically deleted for saving storage
				Base.save_model(current_step)
				logger('**********' * 3 + 'train' + '**********' * 3)
				train_titles, train_values = train_a_ep(config, Base, loader, current_step)
				logger('Time: {};  Step: {};  {}'.format(time_now(), current_step,
														 analyze_names_and_meter(train_titles, train_values)))
				logger('')
				if (current_step) % 3 == 0:
					logger('**********' * 3 + 'test' + '**********' * 3)
					test_titles, test_values, confusion_matrix, metric_values = test_a_ep(config, Base, loader,
																						  current_step)
					logger('Time: {};  Step: {};  {}'.format(time_now(), current_step,
															 analyze_names_and_meter(test_titles, test_values)))
					logger(
						'Time: {};  Step: {}; acc:{}; Precision:{}, Recall:{}, f1:{},Specificity:{}, FPR:{}'.format(
							time_now(),
							current_step,
							metric_values[0], metric_values[1],
							metric_values[2], metric_values[3], metric_values[4], metric_values[5]), '.3f')
					logger(confusion_matrix)
					logger('')

	elif config.mode == 'test':
		logger('**********' * 3 + 'test' + '**********' * 3)
		test_titles, test_values, confusion_matrix, metric_values= test_a_ep(config, Base, loader, start_epoch)
		logger('Time: {};  Step: {};  {}'.format(time_now(), start_epoch,
												 analyze_names_and_meter(test_titles, test_values)))

		logger(
			'Time: {};  Step: {}; acc:{}; Precision:{}, Recall:{}, f1:{}, Specificity:{}, FPR:{}'.format(
				time_now(),
				start_epoch,
				metric_values[0], metric_values[1],
				metric_values[2], metric_values[3], metric_values[4],metric_values[5]), '.3f')
		logger(confusion_matrix)
		logger('')

	elif config.mode == 'localize':
		logger('**********' * 3 + 'localize' + '**********' * 3)
		masks = [os.path.join("./datasets/Localize2/Masks", i) for i in os.listdir("./datasets/Localize2/Masks")]
		masks.sort()

		test_titles, test_values,  = localize_penumonia(config, Base, loader, start_epoch)
		logger('Time: {};  Step: {};  {}'.format(time_now(), start_epoch,
												 analyze_names_and_meter(test_titles, test_values)))
		# logger(confusion_matrix)
		logger('')




if __name__ == '__main__':


	# Configurationtts
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, default='train')

	# output configuration
	parser.add_argument('--save_path', type=str, default='./out', help='path to save models, logs')
	# dataset configuration
	parser.add_argument('--dataset_path', type=str, default='./datasets/covid-19-xray-dataset')

	parser.add_argument('--which_dataset', type=str,
						default='COVID_lungseg')


	parser.add_argument('--Network', type=str, default='ResNet50', help='choose from MAG-SD, ResNet50, ResNet18, InceptionV3, vgg16')
	parser.add_argument('--seg_flag', type=bool, default=False)

	parser.add_argument('--class_num', type=int, default=4, help='identity numbers in training set')
	parser.add_argument('--attention_map_num', type=int, default=32, help='attention map numbers in training set')
	parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
	parser.add_argument('--image_size', type=int, default=224, help='image size for pixel alignment module')
	parser.add_argument('--base_learning_rate', type=float, default=0.001, help='learning rate')
	parser.add_argument('--train_iter', type=int, default=345, help='num of training iteration')
	parser.add_argument('--test_iter', type=int, default=100, help='num of testing iteration')


	# training configuration
	parser.add_argument('--joint_training_steps', type=int, default=150)
	parser.add_argument('--milestones', nargs='+', type=int, default=[75, 125])

	# evaluate configuration
	parser.add_argument('--max_save_model_num', type=int, default=15, help='0 for max num is infinit')

	# parse
	config = parser.parse_args()
	config.milestones = list(np.array(config.milestones))

	main(config)