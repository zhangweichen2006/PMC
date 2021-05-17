MODELPATH="./work_dirs/faster_rcnn_r50_fpn_UCFHMDB_I3d_2cls_Ti3_1_ep12_dec810_nostages_correctcls_CoDANN_-1"
python setup.py develop &
wait
CUDA_VISIBLE_DEVICES=1 python tools/test_curve.py configs/faster_rcnn_r50_fpn_1x_jhmdb.py $MODELPATH --ep 12

# CUDA_VISIBLE_DEVICES=1 python tools/test_curve.py configs/faster_rcnn_r50_fpn_1x_ucf24.py $MODELPATH --ep 12


