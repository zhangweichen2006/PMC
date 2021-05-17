# 3.1.1_u2h_04-08_10w_0.2wt_lr2_lr1_noweight_1ratio_correctw_pre0.2_0.2defaultp_250epoch_1skip_save180200_2Dec2Dec_rightres2_cofidoublespl
# 6.1.1_h2u_04-08_10w_lr15_lr1_noweight_1ratio_correctw_pre0.2_0.2defaultp_250epoch_1skip_save180200_2Dec2Dec_rightres2_cofidoublespl
VER="New_3.1.1_h2u_04-08_10w_lr15_lr1_noweight_1ratio_correctw_pre0.2_0.2defaultp_250epoch_1skip_save180200_2Dec2Dec_rightres2_cofidoublespl_self_atten"
SOURCE="hmdb51_flow_test_split_1"
TARGET="ucf101_flow_test_split_1"
SOURCE2="hmdb51_rgb_test_split_1"
TARGET2="ucf101_rgb_test_split_1"
MODE="Flow"
MODE2="RGB"
FW=0.4
MW=-0.8
LR=0.0015
LR2=0.001
WT=1
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

mkdir -p "./New_txt"

nohup python main.py hmdb51-10 $MODE $MODE2 datalist/$SOURCE.txt datalist/$TARGET.txt datalist/$SOURCE2.txt datalist/$TARGET2.txt --usingTriDec $TD --usingTriDec2 $TD2 --max_pseudo $MP --useT1Only $UT1 --arch BNInception --skip $SKIP --usePrevAcc $USEPREV --defaultPseudoRatio $DEFP --defaultPseudoRatio2 $DEFP2 --pseudo_ratio $PSR --pseudo_ratio2 $PSR2 --usingDoubleDecrease $DD --usingDoubleDecrease2 $DD2 --num_segments 3 --gd 20 --lr $LR --lr2 $LR2  --lr_steps 180 230 --pre_ratio $PRERATIO --epochs $EPOCHS --epochs2 $EPOCHS2 -b $BATCH -j 8 --dropout 0.7 --dropout2 0.8 --snapshot_pref $VER --form_w $FW --main_w $MW --wt $WT --flow_pref $FLOWPREF 2>&1 | tee New_txt/$VER.txt &
