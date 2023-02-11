# Diffusion Recommender Model
This is the pytorch implementation of our paper
> Diffusion Recommender Model

## Environment
- Anaconda 3
- python 3.8.10
- pytorch 1.12.0
- numpy 1.22.3

## Usage
### Data
The experimental data are in './datasets' folder, including Amazon-Book, Yelp and MovieLens-1M.

### Training
#### DiffRec
```
cd ./DiffRec
python main.py --cuda --dataset=$1 --data_path=../datasets/$1/ --lr=$2 --weight_decay=$3 --batch_size=$4 --dims=$5 --emb_size=$6 --mean_type=$7 --steps=$8 --noise_scale=$9 --noise_min=${10} --noise_max=${11} --sampling_steps=${12} --reweight=${13} --log_name=${14} --round=${15} --gpu=${16}
```
or use run.sh
```
cd ./DiffRec
sh run.sh dataset lr weight_decay batch_size dims emb_size mean_type steps noise_scale noise_min noise_max sampling_steps reweight log_name round gpu_id
```

#### L-DiffRec
```
cd ./L-DiffRec
python main.py --cuda --dataset=$1 --data_path=../datasets/$1/ --emb_path=../datasets/ --lr1=$2 --lr2=$3 --wd1=$4 --wd2=$5 --batch_size=$6 --n_cate=$7 --in_dims=$8 --out_dims=$9 --lamda=${10} --mlp_dims=${11} --emb_size=${12} --mean_type=${13} --steps=${14} --noise_scale=${15} --noise_min=${16} --noise_max=${17} --sampling_steps=${18} --reweight=${19} --log_name=${20} --round=${21} --gpu=${22}
```
or use run.sh
```
cd ./L-DiffRec
sh run.sh dataset lr1 lr2 wd1 wd2 batch_size n_cate in_dims out_dims lamda mlp_dims emb_size mean_type steps noise_scale noise_min noise_max sampling_steps reweight log_name round gpu_id
```

#### T-DiffRec
```
cd ./T-DiffRec
python main.py --cuda --dataset=$1 --data_path=../datasets/$1/ --lr=$2 --weight_decay=$3 --batch_size=$4 --dims=$5 --emb_size=$6 --mean_type=$7 --steps=$8 --noise_scale=$9 --noise_min=${10} --noise_max=${11} --sampling_steps=${12} --reweight=${13} --w_min=${14} --w_max=${15} --log_name=${16} --round=${17} --gpu=${18}
```
or use run.sh
```
cd ./T-DiffRec
sh run.sh dataset lr weight_decay batch_size dims emb_size mean_type steps noise_scale noise_min noise_max sampling_steps reweight w_min w_max log_name round gpu_id
```

#### LT-DiffRec
```
cd ./L-DiffRec
python main.py --cuda --dataset=$1 --data_path=../datasets/$1/ --emb_path=../datasets/ --lr1=$2 --lr2=$3 --wd1=$4 --wd2=$5 --batch_size=$6 --n_cate=$7 --in_dims=$8 --out_dims=$9 --lamda=${10} --mlp_dims=${11} --emb_size=${12} --mean_type=${13} --steps=${14} --noise_scale=${15} --noise_min=${16} --noise_max=${17} --sampling_steps=${18} --reweight=${19} --w_min=${20} --w_max=${21} --log_name=${22} --round=${23} --gpu=${24}
```
or use run.sh
```
cd ./L-DiffRec
sh run.sh dataset lr1 lr2 wd1 wd2 batch_size n_cate in_dims out_dims lamda mlp_dims emb_size mean_type steps noise_scale noise_min noise_max sampling_steps reweight w_min w_max log_name round gpu_id
```

### Example: Train DiffRec on Amazon-book under clean setting
```
cd ./DiffRec
sh run.sh amazon-book_clean 5e-5 0 400 [1000] 10 x0 5 0.0001 0.0005 0.005 0 1 log 1 0
```
