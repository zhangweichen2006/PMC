python setup.py develop &
wait
CUDA_VISIBLE_DEVICES=0 python ./tools/train_sl.py ./configs/faster_rcnn_r50_fpn_1x_hmdbucf_ucfpseudo_self_paced.py &

# CUDA_VISIBLE_DEVICES=0 python 7/tools/train.py ./configs/faster_rcnn_r50_fpn_1x_jhmdb.py --work_dir './work_dirs/faster_rcnn_r50_fpn_1x_full_data_jhmdb_I3d_2cls_Ti5_0_ep8_dec57_2stages_correctcls' &

