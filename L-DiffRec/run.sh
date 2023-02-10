nohup python -u main.py --cuda --dataset=$1 --data_path=../datasets/$1/ --emb_path=../datasets/ --lr1=$2 --lr2=$3 --wd1=$4 --wd2=$5 --batch_size=$6 --n_cate=$7 --in_dims=$8 --out_dims=$9 --lamda=${10} --mlp_dims=${11} --emb_size=${12} --mean_type=${13} --steps=${14} --noise_scale=${15} --noise_min=${16} --noise_max=${17} --sampling_steps=${18} --reweight=${19} --log_name=${20} --round=${21} --gpu=${22} > ./log/$1/${21}_$1_$2lr1_$3lr2_$4wd1_$5wd2_bs$6_cate$7_in$8_out$9_lam${10}_dims${11}_emb${12}_${13}_steps${14}_scale${15}_min${16}_max${17}_sample${18}_reweight${19}_${20}.txt 2>&1 &

# sh run.sh amazon-book_clean 5e-4 1e-4 0 0 400 2 [300] [] 0.05 [300] 10 x0 5 0.5 0.001 0.0005 0 1 log 1 0

