MODELPATH="./work_dirs/faster_rcnn_r50_fpn_1x_full_data_stage4_scratch_2x_I3d_2cls_HMDBUCF_UCFHMDB"
python setup.py develop &
wait
# CUDA_VISIBLE_DEVICES=1 python tools/test.py configs/faster_rcnn_r50_fpn_1x_jhmdb.py $MODELPATH/latest.pth --out $MODELPATH/results_u2h.pkl 

CUDA_VISIBLE_DEVICES=1 python tools/test.py configs/faster_rcnn_r50_fpn_1x_ucf24.py $MODELPATH/latest.pth --out $MODELPATH/results.pkl --trim


