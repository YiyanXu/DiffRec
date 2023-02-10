nohup python -u main.py --cuda --dataset=$1 --data_path=../datasets/$1/ --lr=$2 --weight_decay=$3 --batch_size=$4 --dims=$5 --emb_size=$6 --mean_type=$7 --steps=$8 --noise_scale=$9 --noise_min=${10} --noise_max=${11} --sampling_steps=${12} --reweight=${13} --w_min=${14} --w_max=${15} --log_name=${16} --round=${17} --gpu=${18} > ./log/$1/${17}_$1_lr$2_wd$3_bs$4_dims$5_emb$6_$7_steps$8_scale$9_min${10}_max${11}_sample${12}_reweight${13}_wmin${14}_wmax${15}_${16}.txt 2>&1 &

# sh run.sh amazon-book_clean 1e-5 0 400 [1000] 10 x0 5 0.0005 0.001 0.005 0 1 0.1 1.0 log 1 0

























