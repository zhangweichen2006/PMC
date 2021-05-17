import os.path as osp
import torch
import json
import mmcv
import pickle
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
		'Basketball', 'GolfSwing')
# , 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing',
		# 'SalsaSpin','SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling',
		# 'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog')


class UCFJHMDBDataset(Dataset):

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
				 small=False,
				 trim=False,
				 debug=False,
				 da_w=0.1,
				 split=1,
				 revep=0, 
				 pseudo_set=None,
				 self_paced=0,
				 pseudo_test=False,
				 class_balanced=False):
		# path of the data file
		self.self_paced = self_paced
		self.pseudo_set = pseudo_set
		self.class_balanced=class_balanced
		if self.pseudo_set is not None:
			self.pseudo = True
		else:
			self.pseudo = False
		self.pseudo_test = pseudo_test
		self._data_path = osp.join('/home/wzha8158/datasets/Action_DA/','ucf24')
		self.classes = CLASSES
		self.split_idx = split - 1 
		if test_mode:
			self._image_set = 'test'
		else:
			self._image_set = 'train'
		self.ann_info = self._get_ann_file()
		self.eval = self.evaluate_detections
		# filter images with no annotation during training
		self.img_ids, self.img_infos, self.ucf_len, self.hmdb_len = self._filter_imgs(test_mode=test_mode, small=small, trim=trim, pseudo_set=pseudo_set)
		self.set1_len = self.ucf_len
		self.set2_len = self.hmdb_len
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
		self.small = small
		self.trim = trim
		# debug mode or not
		self.debug = debug
		self.revep = revep
		self.da_w = da_w

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

	def _filter_imgs(self, min_size=32, test_mode=False, small=False, trim=False, pseudo_set=None):
		"""Filter images too small or without ground truths."""
		# print("pseudo_set", pseudo_set)
		with open('/home/wzha8158/datasets/Action_DA/jhmdb2/JHMDB-GT.pkl','rb') as fid:
			database = pickle.load(fid)

		if test_mode:
			if small:
				image_set_file = '/home/wzha8158/datasets/Action_DA/ucf24/splitfiles/test_imgs_small.txt'
			elif trim:
				image_set_file = '/home/wzha8158/datasets/Action_DA/ucf24/splitfiles/train_test_imgs_trim.txt'
			else:
				image_set_file = '/home/wzha8158/datasets/Action_DA/ucf24/splitfiles/train_test_imgs.txt'
			
			vids = database['test_videos'][self.split_idx] + database['train_videos'][self.split_idx]
		else:
			image_set_file = '/home/wzha8158/datasets/Action_DA/ucf24/splitfiles/train_test_imgs.txt'

			vids = database['train_videos'][self.split_idx] + database['test_videos'][self.split_idx]

		assert osp.exists(image_set_file), \
		'Path does not exist: {}'.format(image_set_file)
		with open(image_set_file, 'r') as f:
			img_ids = [x.strip() for x in f.readlines()]

		valid_ids = []
		img_infos = []

		valid_ids_rgb = []
		img_infos_rgb = []

		valid_ids_flow = []
		img_infos_flow = []


		if pseudo_set is None:
			for i in img_ids:
				new_i = i.split()
				info = self.ann_info[i]
				if min(info['width'], info['height']) >= min_size:
					valid_ids.append(i)
					info["pseudo_scores"] = [1.]
					info["pseudo_scores_rgb"] = [1.]
					info["pseudo_scores_flow"] = [1.]
					img_infos.append(info)
		else:
			# proportion samples all pseudo labeled target samples to train

			temp_img_infos_dict = {}

			current_epoch = pseudo_set['epoch']
			total_epoch = pseudo_set['total_epoch']
			p_epoch = float(current_epoch) / total_epoch

			ap_strs = []
			num_frames = len(pseudo_set)
			ap_all = np.zeros(len(self.classes), dtype=np.float32)

			best_dict = {}
			best_dict_rgb = {}
			best_dict_flow = {}

			for key, pseu_value in pseudo_set.items():

				cls_pseudo_dict_list = [{'boxes':[],'scores':0}]*len(self.classes)
				top_cls_pseudo_dict_list = [{'box':[],'score':0}]*len(self.classes)

				cls_pseudo_dict_list_rgb = [{'boxes':[],'scores':0}]*len(self.classes)
				top_cls_pseudo_dict_list_rgb = [{'box':[],'score':0}]*len(self.classes)
				
				cls_pseudo_dict_list_flow = [{'boxes':[],'scores':0}]*len(self.classes)
				top_cls_pseudo_dict_list_flow = [{'box':[],'score':0}]*len(self.classes)
				# best_psuedo_dict = {}


				if key in ('total_epoch', 'epoch'):
					continue
				img_path = key

				all_boxes = pseu_value['fuse_pseudo'] #fuse_pseudo
				all_boxes_rgb = pseu_value['pseudo'][0] #fuse_pseudo
				all_boxes_flow = pseu_value['pseudo'][1] #fuse_pseudo

				det_boxes = all_boxes[0]
				det_boxes_rgb = all_boxes_rgb
				det_boxes_flow = all_boxes_flow

				for cls_ind, cls in enumerate(self.classes):
					det_count = 0

					scores = []
					boxes = []

					scores_rgb = []
					boxes_rgb = []

					scores_flow = []
					boxes_flow = []

					# print('box shape', np.array(det_boxes).shape)

					frame_det_boxes = np.copy(det_boxes[cls_ind]) # get frame detections for class cls in nf
					frame_det_boxes_rgb = np.copy(det_boxes_rgb[cls_ind])
					frame_det_boxes_flow = np.copy(det_boxes_flow[cls_ind])
					# cls_gt_boxes = self.get_gt_of_cls(np.copy(gt_boxes[nf]), cls_ind) # get gt boxes for class cls in nf frame
					# num_postives += cls_gt_boxes.shape[0]
					####### Fuse #########
					if frame_det_boxes.shape[0]>0:
						argsort_scores = np.argsort(-frame_det_boxes[:,-1]) # sort in descending order
						for i, k in enumerate(argsort_scores): # start from best scoring detection of cls to end
							box = frame_det_boxes[k, :-1] # detection bounfing box
							score = frame_det_boxes[k,-1] # detection score
							boxes.append(box)
							scores.append(score) # fill score array with score of current detection

						cls_pseudo_dict_list[cls_ind] = {'boxes':boxes,'scores':scores}
						top_cls_pseudo_dict_list[cls_ind] = {'box':boxes[-1],'score':scores[-1]}

					####### RGB #########
					if frame_det_boxes_rgb.shape[0]>0:
						argsort_scores_rgb = np.argsort(-frame_det_boxes_rgb[:,-1]) # sort in descending order
						for i, k in enumerate(argsort_scores_rgb): # start from best scoring detection of cls to end
							box_rgb = frame_det_boxes_rgb[k, :-1] # detection bounfing box
							score_rgb = frame_det_boxes_rgb[k,-1] # detection score
							boxes_rgb.append(box_rgb)
							scores_rgb.append(score_rgb) # fill score array with score of current detection

						cls_pseudo_dict_list_rgb[cls_ind] = {'boxes':boxes,'scores':scores}
						top_cls_pseudo_dict_list_rgb[cls_ind] = {'box':boxes[-1],'score':scores[-1]}

					####### Flow #########
					if frame_det_boxes_flow.shape[0]>0:
						argsort_scores_flow = np.argsort(-frame_det_boxes_flow[:,-1]) # sort in descending order
						for i, k in enumerate(argsort_scores_flow): # start from best scoring detection of cls to end
							box_flow = frame_det_boxes_flow[k, :-1] # detection bounfing box
							score_flow = frame_det_boxes_flow[k,-1] # detection score
							boxes_flow.append(box_flow)
							scores_flow.append(score_flow) # fill score array with score of current detection

						cls_pseudo_dict_list_flow[cls_ind] = {'boxes':boxes,'scores':scores}
						top_cls_pseudo_dict_list_flow[cls_ind] = {'box':boxes[-1],'score':scores[-1]}

					# print("cls_pseudo_dict_list", cls_pseudo_dict_list)
					# print("top_cls_pseudo_dict_list", top_cls_pseudo_dict_list)

				# scores = np.array(scores[:det_count])
				# # print("filter scores", scores)
				# argsort_scores = np.argsort(-scores)
					
				best_score = 0
				best_cls = 0
				best_boxes = np.array([])

				best_score_rgb = 0
				best_cls_rgb = 0
				best_boxes_rgb = np.array([])

				best_score_flow = 0
				best_cls_flow = 0
				best_boxes_flow = np.array([])

				for cls_ind in range(len(self.classes)):
					if top_cls_pseudo_dict_list[cls_ind]['score'] > best_score:
						best_score = top_cls_pseudo_dict_list[cls_ind]['score']
						best_cls = cls_ind
						best_boxes = top_cls_pseudo_dict_list[cls_ind]['box']
				
					if top_cls_pseudo_dict_list_rgb[cls_ind]['score'] > best_score_rgb:
						best_score_rgb = top_cls_pseudo_dict_list_rgb[cls_ind]['score']
						best_cls_rgb = cls_ind
						best_boxes_rgb = top_cls_pseudo_dict_list_rgb[cls_ind]['box']

					if top_cls_pseudo_dict_list_flow[cls_ind]['score'] > best_score_flow:
						best_score_flow = top_cls_pseudo_dict_list_flow[cls_ind]['score']
						best_cls_flow = cls_ind
						best_boxes_flow = top_cls_pseudo_dict_list_flow[cls_ind]['box']

				best_dict[img_path] = best_score
				best_dict_rgb[img_path] = best_score_rgb
				best_dict_flow[img_path] = best_score_flow
				# best_psuedo_dict['boxes'] = best_boxes 
				# best_psuedo_dict['scores'] = best_score 
				# best_psuedo_dict['class'] = best_cls 
				# best_psuedo_dict['img'] = img_path

				img_info = self.ann_info[img_path]
				# print("img_info", img_info)
				img_info["labels"] = [best_cls]
				img_info["labels_rgb"] = [best_cls_rgb]
				img_info["labels_flow"] = [best_cls_flow]
				img_info["pseudo_scores"] = [best_score]
				img_info["pseudo_scores_rgb"] = [best_score_rgb]
				img_info["pseudo_scores_flow"] = [best_score_flow]
				img_info["boxes"] = [best_boxes.tolist()]
				img_info["boxes_rgb"] = [best_boxes_rgb.tolist()]
				img_info["boxes_flow"] = [best_boxes_flow.tolist()]

				temp_img_infos_dict[img_path] = img_info
				# print("valid_ids", valid_ids)
				# print("img_info", img_info)

			
			sorted_list = sorted(best_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
			sorted_list_rgb = sorted(best_dict_rgb.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
			sorted_list_flow = sorted(best_dict_flow.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)


			if self.self_paced != 0:
				select_num = int(self.self_paced * len(sorted_list))
			else:
				select_num = int(0.1 * len(sorted_list))

			if not self.class_balanced:
				selected = sorted_list[:select_num]
				for (k,v) in selected:
					valid_ids.append(k)
					img_infos.append(temp_img_infos_dict[k])

				selected_rgb = sorted_list_rgb[:select_num]
				for (k,v) in selected_rgb:
					valid_ids_rgb.append(k)
					img_infos_rgb.append(temp_img_infos_dict[k])

				selected_flow = sorted_list_flow[:select_num]
				for (k,v) in selected_flow:
					valid_ids_flow.append(k)
					img_infos_flow.append(temp_img_infos_dict[k])
			else:
				class_select_num = int(0.5 * select_num)
				class_balanced_best_dict_list = [0]*len(self.classes)
				for cls_ind in range(len(self.classes)):
					for (k,v) in sorted_list:
						pseudo_class = temp_img_infos_dict[k]['labels'][0]
						if cls_ind == pseudo_class and len(class_balanced_best_dict_list[pseudo_class]) < class_select_num:
							valid_ids.append(k)
							img_infos.append(temp_img_infos_dict[k])
							class_balanced_best_dict_list[pseudo_class] += 1
						elif class_balanced_best_dict_list[pseudo_class] >= class_select_num:
							break
						else:
							continue

				class_balanced_best_dict_list_rgb = [0]*len(self.classes)
				for cls_ind in range(len(self.classes)):
					for (k,v) in sorted_list_rgb:
						pseudo_class = temp_img_infos_dict[k]['labels_rgb'][0]
						if cls_ind == pseudo_class and len(class_balanced_best_dict_list_rgb[pseudo_class]) < class_select_num:
							valid_ids_rgb.append(k)
							img_infos_rgb.append(temp_img_infos_dict[k])
							class_balanced_best_dict_list_rgb[pseudo_class] += 1
						elif class_balanced_best_dict_list_rgb[pseudo_class] >= class_select_num:
							break
						else:
							continue

				class_balanced_best_dict_list_flow = [0]*len(self.classes)
				for cls_ind in range(len(self.classes)):
					for (k,v) in sorted_list_flow:
						pseudo_class = temp_img_infos_dict[k]['labels_flow'][0]
						if cls_ind == pseudo_class and len(class_balanced_best_dict_list_flow[pseudo_class]) < class_select_num:
							valid_ids_flow.append(k)
							img_infos_flow.append(temp_img_infos_dict[k])
							class_balanced_best_dict_list_flow[pseudo_class] += 1
						elif class_balanced_best_dict_list_flow[pseudo_class] >= class_select_num:
							break
						else:
							continue

			# print("valid_ids", valid_ids)
			# info = {}
			# valid_ids.append(img_path)
			# info['height'], info['width'] = pseu_value['height'], pseu_value['width']
			# boxes = []
			# labels = []

			# boxes = np.concatenate(boxes,0)
			# labels = np.concatenate(labels,0)
			# info['boxes'] = boxes
			# info['labels'] = labels
			# img_infos.append(info)


		ucf_len = len(valid_ids)
		print("ucf_len",ucf_len)

		root_path = '/home/wzha8158/datasets/Action_DA/jhmdb2/Frames'
		# img_ids = []
		# valid_ids = []
		# img_infos = []
		for vid in vids:
			nframe = database['nframes'][vid]
			for f in range(1, nframe+1):
				info = {}
				img_path = osp.join(root_path, vid, '{:05d}.png'.format(f))
				valid_ids.append(img_path)
				info['height'], info['width'] = database['resolution'][vid]
				boxes = []
				labels = []
				for k, v in database['gttubes'][vid].items():
					for bb in v:
						try:
							boxes.append(bb[f-1,1:].reshape(1,-1))
						except:
							print(nframe, vid, bb)
							raise(Exception)
						labels.append(np.array([k]).reshape(-1))
				boxes = np.concatenate(boxes,0)
				labels = np.concatenate(labels,0)
				info['boxes'] = boxes
				info['labels'] = labels
				img_infos.append(info)

		hmdb_len = len(valid_ids) - ucf_len 
		print("hmdb_len",hmdb_len)

		return valid_ids, img_infos, ucf_len, hmdb_len

	def _get_ann_file(self):
		prefix = self._image_set
		if prefix in ['test01', 'test02', 'test03']:
			prefix = prefix[:-2]
		ann_path = osp.join(self._data_path, 'splitfiles', \
		'ucf24_annotation_' + 'train' +'.json')
		with open(ann_path, 'r') as f:
			ann = json.load(f)

		ann_path2 = osp.join(self._data_path, 'splitfiles', \
		'ucf24_annotation_' + 'test' +'.json')
		with open(ann_path2, 'r') as f2:
			ann2 = json.load(f2)

		ann3 = {**ann, **ann2}
		return ann3

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
		# print("ann_info")
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
			if ann_info['labels'][i] + 1 not in [i for i in range(len(self.classes)+1)]:
				gt_labels.append(0)
			else:
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

		if 'pseudo_scores' in ann_info:
			pseudo_scores = np.array(ann_info['pseudo_scores'], dtype=np.float32)
			pseudo_scores_rgb = np.array(ann_info['pseudo_scores_rgb'], dtype=np.float32)
			pseudo_scores_flow = np.array(ann_info['pseudo_scores_flow'], dtype=np.float32)
		else:
			pseudo_scores = np.array([0])
			pseudo_scores_rgb = np.array([0])
			pseudo_scores_flow = np.array([0])
		ann = dict(
			bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore, pseudo_scores=pseudo_scores,  pseudo_scores_rgb=pseudo_scores_rgb, pseudo_scores_flow=pseudo_scores_flow)

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
		if i < self.ucf_len:
			aff = '.jpg'
		else:
			aff = '.png'

		for k in range(0,11):
			info = image_path.split('/')
			num = int(info[-1].rstrip(aff))
			num = num + k
			path_k = '/'.join(info[:-1]) + '/{:05d}'.format(num)+aff
			if osp.exists(path_k):
				image_paths.append(path_k)
			else:
				image_paths.append(image_paths[-1])
		for k in range(0,9):
			info = image_path.split('/')
			num = int(info[-1].rstrip(aff))
			num = num - k
			path_k = '/'.join(info[:-1]) + '/{:05d}'.format(num)+aff
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

			if idx < self.ucf_len:
				dsets = 'ucf24'
				flowprev = '_'
			else:
				dsets = 'jhmdb2'
				flowprev = ''

			for k, path in enumerate(image_paths):
				info = path.split('/')
				if idx < self.ucf_len:
					flowaff = info[-1]
				else:
					flowaff = info[-1].strip('.png') + '.jpg'
				path_x = '/home/wzha8158/datasets/Action_DA/'+dsets+'/flownet2-images/' + info[-3] + '/' + info[-2] + '/flow_x'+flowprev + flowaff
				path_y = '/home/wzha8158/datasets/Action_DA/'+dsets+'/flownet2-images/' + info[-3] + '/' + info[-2] + '/flow_y'+flowprev + flowaff
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
			if self.pseudo_set is not None:
				pseudo_bboxes_scores = ann['pseudo_scores']
				pseudo_bboxes_scores_rgb = ann['pseudo_scores_rgb']
				pseudo_bboxes_scores_flow = ann['pseudo_scores_flow']
			else:
				pseudo_bboxes_scores = np.array([1.])
				pseudo_bboxes_scores_rgb = np.array([1.])
				pseudo_bboxes_scores_flow = np.array([1.])
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

			if idx < self.ucf_len:
				for img in imgs:
					img, _, _, _ = self.img_transform(
						img, img_scale, flip, True, False)
					data_imgs.append(img)
				for img in flow_imgs:
					img, _, _, _ = self.img_transform(
						img, img_scale, flip, False, True)
					data_imgs.append(img)
			else:
				for img in imgs:
					img, _, _, _ = self.img_transform(
						img, img_scale, flip, False, False, is_caffe=False, S3DG_data=True)
					data_imgs.append(img)

				for img in flow_imgs:
					img, _, _, _ = self.img_transform(
						img, img_scale, flip, False, True, is_caffe=False, S3DG_data=True)
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
				filename=img_info['filename'],
				flip=flip)


			# all_index = []
			# for i in range(1, len(self.classes)+1):
			#     # print("pos_gt_labels_flow[i].", pos_gt_labels_flow[i])
			#     index = (gt_labels[0] == i).nonzero().tolist()
			#     if index is not None:
			#         all_index = all_index.extend(index)

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
			data['src_dom_label'] = DC(torch.FloatTensor([[1.]]))
			data['tgt_dom_label'] = DC(torch.FloatTensor([[0.]]))
			data['datasetlen'] = DC(min(self.ucf_len, self.hmdb_len), cpu_only=True)
			data['pseudo'] = DC(self.pseudo, cpu_only=True)
			# print("to_tensor(pseudo_bboxes_scores)",to_tensor(pseudo_bboxes_scores))
			data['pseu_score'] = DC(to_tensor(pseudo_bboxes_scores))
			data['pseu_score_rgb'] = DC(to_tensor(pseudo_bboxes_scores_rgb))
			data['pseu_score_flow'] = DC(to_tensor(pseudo_bboxes_scores_flow))

			# data['da_weights'] = DC(self.da_w, cpu_only=True)

			return data

	def prepare_test_img(self, idx):
		"""Prepare an image for testing (multi-scale and flipping)"""
		img_info = self.img_infos[idx]
		img_info['filename'] = self.img_ids[idx]
		image_paths = self.image_path_at(idx)
		rgb_imgs = [mmcv.imread(path) for path in image_paths]

		flow_imgs = []

		if idx < self.ucf_len:
			dsets = 'ucf24'
			flowprev = '_'
			is_hmdb=False
		else:
			dsets = 'jhmdb2'
			flowprev = ''
			is_hmdb=True

		for k, path in enumerate(image_paths):
			info = path.split('/')
			if idx < self.ucf_len:
				flowaff = info[-1]
			else:
				flowaff = info[-1].strip('.png') + '.jpg'
			path_x = '/home/wzha8158/datasets/Action_DA/'+dsets+'/flownet2-images/' + info[-3] + '/' + info[-2] + '/flow_x'+flowprev + flowaff
			path_y = '/home/wzha8158/datasets/Action_DA/'+dsets+'/flownet2-images/' + info[-3] + '/' + info[-2] + '/flow_y'+flowprev + flowaff

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

		def prepare_single(img, scale, flip, proposal=None, I3D_data=False, is_flow=False, is_caffe=False, S3DG_data=False):
			_img, img_shape, pad_shape, scale_factor = self.img_transform(
				img, scale, flip, I3D_data, is_flow, is_caffe=False, S3DG_data=False)
			_img = to_tensor(_img).float()
			_img_meta = dict(
				ori_shape=(img_info['height'], img_info['width'], 3),
				img_shape=img_shape,
				pad_shape=pad_shape,
				filename=img_info['filename'],
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
				img, scale, False, proposal, not is_hmdb, False, is_caffe, S3DG_data)
				imgs.append(_img)
			for img in flow_imgs:
				_img, _, _ = prepare_single(
				img, scale, False, proposal, False, True, is_caffe, S3DG_data)
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
			ann = self.img_infos[index]
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

		type_list = ['rgb:', 'flow:', 'stage1:', 'stage2:', 'stage3:','stage4']

		for n in range(len(all_boxes[0])):
			print(type_list[n])
			temp_boxes = []
			for nf_boxes in all_boxes:
				temp_boxes.append(nf_boxes[n])
			mAP, ap_all, ap_strs = ucf24_evaluate_detections(gt, temp_boxes, CLASSES=CLASSES, iou_thresh=0.5)


			for ap_str in ap_strs:
				print(ap_str)
			ptr_str = '\nMEANAP:::=>'+str(mAP)+'\n'
			print(ptr_str)


