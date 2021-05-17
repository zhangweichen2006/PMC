import os.path as osp
import torch
import json
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from scipy.misc import imread

from .transforms import (ImageTransform, BboxTransform, MaskTransform,
						 Numpy2Tensor)
from .utils import to_tensor, show_ann, random_scale

from .evaluation import ucf24_evaluate_detections

CLASSES = (  # always index 0
		'Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing',
		'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing',
		'SalsaSpin','SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling',
		'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog')


class UCF24Dataset(Dataset):

	def __init__(self,
				 ann_file,
				 img_prefix,
				 img_scale,
				 img_norm_cfg,
				 size_divisor=None,
				 proposal_file=None,
				 num_max_proposals=1000,
				 flip_ratio=0,
				 with_mask=True,
				 with_crowd=True,
				 with_label=True,
				 test_mode=False,
				 debug=False):
		# path of the data file
		self._data_path = osp.join('/home/rusu5516/JAAD_dataset','ucf24')
		if test_mode:
			self._image_set = 'test'
		else:
			self._image_set = 'train'
		self.ann_info = self._get_ann_file()
		self.eval = self.evaluate_detections
		# filter images with no annotation during training
		self.img_ids, self.img_infos = self._filter_imgs(test_mode=test_mode)
		# else:
		# 	self.img_ids = self.coco.getImgIds()
		# 	self.img_infos = [
		# 		self.coco.loadImgs(idx)[0] for idx in self.img_ids
		# 	]
		assert len(self.img_ids) == len(self.img_infos)
		# get the mapping from original category ids to labels
		# self.cat_ids = self.coco.getCatIds()
		# self.cat2label = {
		# 	cat_id: i + 1
		# 	for i, cat_id in enumerate(self.cat_ids)
		# }
		# prefix of images path
		self.img_prefix = img_prefix
		# (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
		self.img_scales = img_scale if isinstance(img_scale,
												  list) else [img_scale]
		assert mmcv.is_list_of(self.img_scales, tuple)
		# color channel order and normalize configs
		self.img_norm_cfg = img_norm_cfg
		# proposals
		# TODO: revise _filter_imgs to be more flexible
		if proposal_file is not None:
			self.proposals = mmcv.load(proposal_file)
			ori_ids = self.coco.getImgIds()
			sorted_idx = [ori_ids.index(id) for id in self.img_ids]
			self.proposals = [self.proposals[idx] for idx in sorted_idx]
		else:
			self.proposals = None
		self.num_max_proposals = num_max_proposals
		# flip ratio
		self.flip_ratio = flip_ratio
		assert flip_ratio >= 0 and flip_ratio <= 1
		# padding border to ensure the image size can be divided by
		# size_divisor (used for FPN)
		self.size_divisor = size_divisor
		# with crowd or not, False when using RetinaNet
		self.with_crowd = with_crowd
		# with mask or not
		self.with_mask = with_mask
		# with label is False for RPN
		self.with_label = with_label
		# in test mode or not
		self.test_mode = test_mode
		# debug mode or not
		self.debug = debug

		# set group flag for the sampler
		self._set_group_flag()
		# transforms
		self.img_transform = ImageTransform(
			size_divisor=self.size_divisor, **self.img_norm_cfg)
		self.bbox_transform = BboxTransform()
		self.mask_transform = MaskTransform()
		self.numpy2tensor = Numpy2Tensor()

	def __len__(self):
		return len(self.img_ids)

	def _filter_imgs(self, min_size=32, test_mode=False):
		"""Filter images too small or without ground truths."""
		if test_mode:
			image_set_file = '/home/rusu5516/JAAD_dataset/ucf24/splitfiles/test_imgs.txt'
		else:
			image_set_file = '/home/rusu5516/JAAD_dataset/ucf24/splitfiles/train_imgs.txt'

		assert osp.exists(image_set_file), \
		'Path does not exist: {}'.format(image_set_file)
		with open(image_set_file, 'r') as f:
			img_ids = [x.strip() for x in f.readlines()]
		# img_ids = img_ids[::10]
		valid_ids = []
		img_infos = []
		for i in img_ids:
			info = self.ann_info[i]
			if min(info['width'], info['height']) >= min_size:
				valid_ids.append(i)
				img_infos.append(info)
		return valid_ids, img_infos

	def _get_ann_file(self):
		prefix = self._image_set
		if prefix in ['test01', 'test02', 'test03']:
			prefix = prefix[:-2]
		ann_path = osp.join(self._data_path, 'splitfiles', \
		'ucf24_annotation_' + prefix +'.json')
		with open(ann_path, 'r') as f:
			ann = json.load(f)
		return ann

	def _load_ann_info(self, idx):
		img_id = self.img_ids[idx]
		ann_ids = self.coco.getAnnIds(imgIds=img_id)
		ann_info = self.coco.loadAnns(ann_ids)
		return ann_info

	def _parse_ann_info(self, ann_info, with_mask=True):
		"""Parse bbox and mask annotation.

		Args:
			ann_info (list[dict]): Annotation info of an image.
			with_mask (bool): Whether to parse mask annotations.

		Returns:
			dict: A dict containing the following keys: bboxes, bboxes_ignore,
				labels, masks, mask_polys, poly_lens.
		"""
		gt_bboxes = []
		gt_labels = []
		gt_bboxes_ignore = []
		# Two formats are provided.
		# 1. mask: a binary map of the same size of the image.
		# 2. polys: each mask consists of one or several polys, each poly is a
		# list of float.
		num_obj = len(ann_info['labels'])
		if with_mask:
			gt_masks = []
			gt_mask_polys = []
			gt_poly_lens = []
		for i in range(num_obj):
			# if ann.get('ignore', False):
			#     continue
			x1, y1, x2, y2 = ann_info['boxes'][i]
			# if ann['area'] <= 0 or w < 1 or h < 1:
			#     continue
			bbox = [x1, y1, x2, y2]
			# if ann['iscrowd']:
			#     gt_bboxes_ignore.append(bbox)
			# else:
			gt_bboxes.append(bbox)
			gt_labels.append(ann_info['labels'][i] + 1)
			if with_mask:
				gt_masks.append(self.coco.annToMask(ann))
				mask_polys = [
					p for p in ann['segmentation'] if len(p) >= 6
				]  # valid polygons have >= 3 points (6 coordinates)
				poly_lens = [len(p) for p in mask_polys]
				gt_mask_polys.append(mask_polys)
				gt_poly_lens.extend(poly_lens)
		if gt_bboxes:
			gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
			gt_labels = np.array(gt_labels, dtype=np.int64)
		else:
			gt_bboxes = np.zeros((0, 4), dtype=np.float32)
			gt_labels = np.array([0], dtype=np.int64)

		if gt_bboxes_ignore:
			gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
		else:
			gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

		ann = dict(
			bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

		if with_mask:
			ann['masks'] = gt_masks
			# poly format is not used in the current implementation
			ann['mask_polys'] = gt_mask_polys
			ann['poly_lens'] = gt_poly_lens
		return ann

	def _set_group_flag(self):
		"""Set flag according to image aspect ratio.

		Images with aspect ratio greater than 1 will be set as group 1,
		otherwise group 0.
		"""
		self.flag = np.zeros(len(self.img_ids), dtype=np.uint8)
		for i in range(len(self.img_ids)):
			img_info = self.img_infos[i]
			if img_info['width'] / img_info['height'] > 1:
				self.flag[i] = 1

	def _rand_another(self, idx):
		pool = np.where(self.flag == self.flag[idx])[0]
		return np.random.choice(pool)

	def image_path_at(self, i):
		image_paths = []
		image_path = self.img_ids[i]
		for k in range(0,11):
			info = image_path.split('/')
			num = int(info[-1].rstrip('.jpg'))
			num = num + k
			path_k = '/'.join(info[:-1]) + '/{:05d}.jpg'.format(num)
			if osp.exists(path_k):
				image_paths.append(path_k)
			else:
				image_paths.append(image_paths[-1])
		for k in range(0,9):
			info = image_path.split('/')
			num = int(info[-1].rstrip('.jpg'))
			num = num - k
			path_k = '/'.join(info[:-1]) + '/{:05d}.jpg'.format(num)
			if osp.exists(path_k):
				image_paths = [path_k]+image_paths
			else:
				image_paths = [image_paths[0]] + image_paths
		assert osp.exists(image_path), \
		'Path does not exist: {}'.format(image_path)
		return image_paths

	def __getitem__(self, idx):
		if self.test_mode:
			return self.prepare_test_img(idx)
		while True:
			img_info = self.img_infos[idx]
			# ann_info = self._load_ann_info(idx)

			# load image
			img_info['filename'] = self.img_ids[idx]
			image_paths = self.image_path_at(idx)
			imgs = [mmcv.imread(path) for path in image_paths]

			flow_imgs = []

			for k, path in enumerate(image_paths):
				info = path.split('/')
				path_x = '/home/rusu5516/JAAD_dataset/ucf24/flownet2-images/' + info[-3] + '/' + info[-2] + '/flow_x_' + info[-1]
				path_y = '/home/rusu5516/JAAD_dataset/ucf24/flownet2-images/' + info[-3] + '/' + info[-2] + '/flow_y_' + info[-1]
				im_x = imread(path_x, mode='L')
				im_y = imread(path_y, mode='L')

				im_x = im_x[:,:,np.newaxis]
				im_y = im_y[:,:,np.newaxis]

				im = np.concatenate((im_x,im_y,im_y), axis=2)

				flow_imgs.append(im)

			if self.debug:
				show_ann(self.coco, img, ann_info)

			# load proposals if necessary
			if self.proposals is not None:
				proposals = self.proposals[idx][:self.num_max_proposals]
				# TODO: Handle empty proposals properly. Currently images with
				# no proposals are just ignored, but they can be used for
				# training in concept.
				if len(proposals) == 0:
					idx = self._rand_another(idx)
					continue
				if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
					raise AssertionError(
						'proposals should have shapes (n, 4) or (n, 5), '
						'but found {}'.format(proposals.shape))
				if proposals.shape[1] == 5:
					scores = proposals[:, 4]
					proposals = proposals[:, :4]
				else:
					scores = None

			ann = self._parse_ann_info(img_info, self.with_mask)
			gt_bboxes = ann['bboxes']
			gt_labels = ann['labels']
			gt_bboxes_ignore = ann['bboxes_ignore']
			# skip the image if there is no valid gt bbox
			if len(gt_bboxes) == 0:
				idx = self._rand_another(idx)
				continue

			# apply transforms
			flip = True if np.random.rand() < self.flip_ratio else False
			img_scale = random_scale(self.img_scales)  # sample a scale
			data_imgs = []
			img, img_shape, pad_shape, scale_factor = self.img_transform(
					imgs[9], img_scale, flip)
			data_imgs.append(img)
			for img in imgs:
				img, _, _, _ = self.img_transform(
					img, img_scale, flip, True, False)
				data_imgs.append(img)
			for img in flow_imgs:
				img, _, _, _ = self.img_transform(
					img, img_scale, flip, False, True)
				data_imgs.append(img)
			data_imgs = np.concatenate(data_imgs, axis=0)
			if self.proposals is not None:
				proposals = self.bbox_transform(proposals, img_shape,
												scale_factor, flip)
				proposals = np.hstack([proposals, scores[:, None]
									   ]) if scores is not None else proposals
			gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
											flip)
			gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
												   scale_factor, flip)

			if self.with_mask:
				gt_masks = self.mask_transform(ann['masks'], pad_shape,
											   scale_factor, flip)

			ori_shape = (img_info['height'], img_info['width'], 3)
			img_meta = dict(
				ori_shape=ori_shape,
				img_shape=img_shape,
				pad_shape=pad_shape,
				scale_factor=scale_factor,
				flip=flip)

			data = dict(
				img=DC(to_tensor(data_imgs).float(), stack=True),
				img_meta=DC(img_meta, cpu_only=True),
				gt_bboxes=DC(to_tensor(gt_bboxes)))
			if self.proposals is not None:
				data['proposals'] = DC(to_tensor(proposals))
			if self.with_label:
				data['gt_labels'] = DC(to_tensor(gt_labels))
			if self.with_crowd:
				data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
			if self.with_mask:
				data['gt_masks'] = DC(gt_masks, cpu_only=True)
			return data

	def prepare_test_img(self, idx):
		"""Prepare an image for testing (multi-scale and flipping)"""
		img_info = self.img_infos[idx]
		img_info['filename'] = self.img_ids[idx]
		image_paths = self.image_path_at(idx)
		rgb_imgs = [mmcv.imread(path) for path in image_paths]

		flow_imgs = []

		for k, path in enumerate(image_paths):
			info = path.split('/')
			path_x = '/home/rusu5516/JAAD_dataset/ucf24/flownet2-images/' + info[-3] + '/' + info[-2] + '/flow_x_' + info[-1]
			path_y = '/home/rusu5516/JAAD_dataset/ucf24/flownet2-images/' + info[-3] + '/' + info[-2] + '/flow_y_' + info[-1]
			im_x = imread(path_x, mode='L')
			im_y = imread(path_y, mode='L')

			im_x = im_x[:,:,np.newaxis]
			im_y = im_y[:,:,np.newaxis]

			im = np.concatenate((im_x,im_y,im_y), axis=2)

			flow_imgs.append(im)
		# img = mmcv.imread(osp.join(self.img_prefix, img_info['file_name']))
		if self.proposals is not None:
			proposal = self.proposals[idx][:self.num_max_proposals]
			if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
				raise AssertionError(
					'proposals should have shapes (n, 4) or (n, 5), '
					'but found {}'.format(proposal.shape))
		else:
			proposal = None

		def prepare_single(img, scale, flip, proposal=None, I3D_data=False, is_flow=False):
			_img, img_shape, pad_shape, scale_factor = self.img_transform(
				img, scale, flip, I3D_data, is_flow)
			_img = to_tensor(_img).float()
			_img_meta = dict(
				ori_shape=(img_info['height'], img_info['width'], 3),
				img_shape=img_shape,
				pad_shape=pad_shape,
				scale_factor=scale_factor,
				flip=flip)
			if proposal is not None:
				if proposal.shape[1] == 5:
					score = proposal[:, 4]
					proposal = proposal[:, :4]
				else:
					score = None
				_proposal = self.bbox_transform(proposal, img_shape,
												scale_factor, flip)
				_proposal = np.hstack([_proposal, score[:, None]
									   ]) if score is not None else _proposal
				_proposal = to_tensor(_proposal)
			else:
				_proposal = None
			return _img, _img_meta, _proposal

		imgs = []
		img_metas = []
		proposals = []
		for scale in self.img_scales:
			_img, _img_meta, _proposal = prepare_single(
				rgb_imgs[9], scale, False, proposal)
			imgs.append(_img)
			img_metas.append(DC(_img_meta, cpu_only=True))
			proposals.append(_proposal)
			for img in rgb_imgs:
				_img, _, _ = prepare_single(
				img, scale, False, proposal, True, False)
				imgs.append(_img)
			for img in flow_imgs:
				_img, _, _ = prepare_single(
				img, scale, False, proposal, False, True)
				imgs.append(_img)
			# if self.flip_ratio > 0:
			# 	_img, _img_meta, _proposal = prepare_single(
			# 		img, scale, True, proposal)
			# 	imgs.append(_img)
			# 	img_metas.append(DC(_img_meta, cpu_only=True))
			# 	proposals.append(_proposal)
			imgs = [torch.cat(imgs,0)]
			# print(imgs[0].size())
		data = dict(img=imgs, img_meta=img_metas)
		if self.proposals is not None:
			data['proposals'] = proposals
		return data

	def evaluate_detections(self, all_boxes, output_dir=None):
		gt = []
		for index in self.img_ids:
		    ann = self.ann_info[index]
		    width = ann['width']
		    height = ann['height']
		    bboxes = ann['boxes']
		    labels = ann['labels']
		    num_objs = len(ann['labels'])
		    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
		    gt_classes = np.zeros((num_objs), dtype=np.int32)

		    for idx, bbox in enumerate(bboxes):
		        x1 = np.max((0, bbox[0]))
		        y1 = np.max((0, bbox[1]))
		        x2 = np.min((width - 1, bbox[2] ))
		        y2 = np.min((height - 1, bbox[3] - 1))
		        boxes[idx,:] = [x1, y1, x2, y2]
		        gt_classes[idx] = labels[idx]
		    gt_boxes = np.empty((boxes.shape[0], 5), dtype=np.float32)
		    gt_boxes[:, 0:4] = boxes
		    gt_boxes[:, 4] = gt_classes
		    gt.append(gt_boxes)
		mAP, ap_all, ap_strs = ucf24_evaluate_detections(gt, all_boxes, CLASSES=CLASSES, iou_thresh=0.5)


		for ap_str in ap_strs:
		    print(ap_str)
		ptr_str = '\nMEANAP:::=>'+str(mAP)+'\n'
		print(ptr_str)
