VER="5.20.0_u2o_000103_10w_0.2wt_lr2_lr1_noweight_1ratio_correctw_pre0.2_0.2defaultp_250epoch_1skip_save180200_2Dec2Dec_CoSPL_LearnCD_10lrC0.1lrD_D1revD2"
SOURCE="olympic6_rgb_test_split_1"
TARGET="ucf6_rgb_train_split_1"
SOURCE2="olympic6_rgb_test_split_2"
TARGET2="ucf6_rgb_train_split_2"
MODE="Flow"
MODE2="RGB"
FW=0.0
MW=-0.1
FW2=0.0
MW2=-0.3
LR=0.002
LR2=0.001
WT=0.2
BATCH=32
EPOCHS=250
EPOCHS2=180
PRERATIO=0.2
FLOWPREF="flow_"
PSR=1
PSR2=1
DD="True"
DD2="True"
TD="False"
TD2="False"
USEPREV="False"
DEFP=0.2
DEFP2=0.2
SKIP=1
MP=1
UT1="False"
nohup python main_cospcan_ratio_2Dec_prev_save_cospl_learnCD.py ucf6 $MODE $MODE2 datalist/$SOURCE.txt datalist/$TARGET.txt datalist/$SOURCE2.txt datalist/$TARGET2.txt --usingTriDec $TD --usingTriDec2 $TD2 --max_pseudo $MP --useT1Only $UT1 --arch BNInception --skip $SKIP --usePrevAcc $USEPREV --defaultPseudoRatio $DEFP --defaultPseudoRatio2 $DEFP2 --pseudo_ratio $PSR --pseudo_ratio2 $PSR2 --usingDoubleDecrease $DD --usingDoubleDecrease2 $DD2 --num_segments 3 --gd 20 --lr $LR --lr2 $LR2  --lr_steps 180 230 --pre_ratio $PRERATIO --epochs $EPOCHS --epochs2 $EPOCHS2 -b $BATCH -j 8 --dropout 0.7 --dropout2 0.8 --snapshot_pref $VER --form_w $FW --main_w $MW  --form_w2 $FW2 --main_w2 $MW2 --wt $WT --flow_pref $FLOWPREF 2>&1 | tee $VER.txt &
# VER="3.2_u2h_04-08_10w_lr2_noweight_0.5ratio_correctw_0.2defaultp_250epoch_1skip_save180200_T1Only_prev_3Dec"
# SOURCE="ucf101_flow_test_split_1"
# TARGET="hmdb51_flow_test_split_1"
# MODE="Flow"
# SMODE="flow"
# FW=0.4
# MW=-0.8
# LR=0.002
# GPU=1
# WT=1
# BATCH=48
# EPOCHS=250
# PRERATIO=0.2
# FLOWPREF="flow_"
# PSR=0.5
# DD="True"
# TD="True"
# USEPREV="True"
# DEFP=0.2
# SKIP=1
# MP=0.5
# UT1="True"
# echo "${VER}_${SMODE}"
# nohup python main_spcan_ratio_2Dec_prev_save.py hmdb51-10 $MODE datalist/$SOURCE.txt datalist/$TARGET.txt --usingTriDec $TD --max_pseudo $MP --useT1Only $UT1 --arch BNInception --skip $SKIP --usePrevAcc $USEPREV --defaultPseudoRatio $DEFP --pseudo_ratio $PSR --usingDoubleDecrease $DD --num_segments 3 --gd 20 --lr $LR --lr_steps 180 230 --pre_ratio $PRERATIO --epochs $EPOCHS -b $BATCH -j 8 --dropout 0.8 --snapshot_pref $VER --form_w $FW --main_w $MW --gpu $GPU --wt $WT --flow_pref $FLOWPREF > $VER.txt &
# VER="3.3_u2h_04-08_10w_lr2_noweight_1ratio_correctw_3Dec_0.1defaultp_250epoch_1skip_save180200"
# SOURCE="ucf101_flow_test_split_1"
# TARGET="hmdb51_flow_test_split_1"
# MODE="Flow"
# SMODE="flow"
# FW=0.4
# MW=-0.8
# LR=0.002
# GPU=2
# WT=1
# BATCH=48
# EPOCHS=250
# PRERATIO=0.2
# FLOWPREF="flow_"
# PSR=1
# DD="True"
# TD="True"
# USEPREV="False"
# DEFP=0.1
# SKIP=1
# MP=0.5
# UT1="False"
# echo "${VER}_${SMODE}"
# nohup python main_spcan_ratio_2Dec_prev_save.py hmdb51-10 $MODE datalist/$SOURCE.txt datalist/$TARGET.txt --usingTriDec $TD --max_pseudo $MP --useT1Only $UT1 --arch BNInception --skip $SKIP --usePrevAcc $USEPREV --defaultPseudoRatio $DEFP --pseudo_ratio $PSR --usingDoubleDecrease $DD --num_segments 3 --gd 20 --lr $LR --lr_steps 180 230 --pre_ratio $PRERATIO --epochs $EPOCHS -b $BATCH -j 8 --dropout 0.8 --snapshot_pref $VER --form_w $FW --main_w $MW --gpu $GPU --wt $WT --flow_pref $FLOWPREF > $VER.txt &