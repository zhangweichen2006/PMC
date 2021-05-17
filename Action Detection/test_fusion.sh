MODELPATH=./work_dirs/faster_rcnn_r50_fpn_1x_full_data_jhmdb_I3d_2cls_Ti5_3_ep8_dec57_nostage_correctcls
FUSEPATH=./work_dirs/faster_rcnn_r50_fpn_1x_full_data_jhmdb_I3d_2cls_Ti5_3_ep8_dec57_nostage_correctcls/results.pkl
python setup.py develop &
wait
CUDA_VISIBLE_DEVICES=1 python tools/fusion_test.py configs/faster_rcnn_r50_fpn_1x_jhmdb.py $MODELPATH/latest.pth $FUSEPATH --stages [0,1]

# CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/faster_rcnn_r50_fpn_1x_ucf24.py 
