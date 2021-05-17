# rm -rf ./txt &
# wait
# mkdir txt &
# wait
# rm -rf ./graph &
# wait
# rm -f discriminative_dann.pyc &
# wait
# nohup python b2w.py > ./txt/b2w.txt &
# nohup python w2l.py > ./txt/w2l.txt &
# nohup python w2b.py > ./txt/w2b.txt &
# nohup python rgbd10_RGBDepth.py --source_set 'Bremen' --target_set 'Washington' --gpu '0' --batch_size 16 --base_lr 0.0003 --num_class 10\
#                        --pretrain_sample 30000 --train_sample 120000 --form_w 0.0 --main_w -0.3 --form_w2 0.0 --main_w2 -0.3 --wp 0.055 --lrG 0.003 --lrD 0.0003 --nesterov False > ./txt/b2w_303.txt &

nohup python rgbd10_RGBDepth.py --source_set 'Washington' --target_set 'Bremen' --gpu '0' --batch_size 16 --base_lr 0.0003 --num_class 10\
                       --pretrain_sample 30000 --train_sample 120000 --form_w 0.0 --main_w -0.3 --form_w2 0.0 --main_w2 -0.3 --wp 0.055 --lrG 0.003 --lrD 0.0003 --nesterov False > ./txt/w2b_303.txt &

nohup python rgbd10_RGBDepth.py --source_set 'Washington' --target_set 'Caltech' --gpu '1' --batch_size 16 --base_lr 0.0003 --num_class 10\
                       --pretrain_sample 30000 --train_sample 120000 --form_w 0.0 --main_w -0.3 --form_w2 0.0 --main_w2 -0.3 --wp 0.055 --lrG 0.003 --lrD 0.0003 --nesterov False > ./txt/w2c_303.txt &

nohup python rgbd10_RGBDepth.py --source_set 'Bremen' --target_set 'Caltech' --gpu '2' --batch_size 16 --base_lr 0.0003 --num_class 10\
                       --pretrain_sample 30000 --train_sample 120000 --form_w 0.0 --main_w -0.3 --form_w2 0.0 --main_w2 -0.3 --wp 0.055 --lrG 0.003 --lrD 0.0003 --nesterov False > ./txt/b2c_303.txt &

# nohup python rgbd10_RGBDepth.py --source_set 'Latina' --target_set 'Washington' --gpu '0' --batch_size 16 --base_lr 0.0003 --num_class 10\
#                        --pretrain_sample 150000 --train_sample 600000 --form_w 0.0 --main_w -0.3 --wp 0.055 --nesterov False > ./txt/l2w.txt &
# nohup python rgbd10_RGBDepth.py --source_set 'Washington' --target_set 'Latina' --gpu '2' --batch_size 16 --base_lr 0.0003 --num_class 10\
#                        --pretrain_sample 100000 --train_sample 400000 --form_w 0.0 --main_w -0.3 --wp 0.055 --nesterov False > ./txt/w2l.txt &