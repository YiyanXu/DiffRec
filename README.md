# Diffusion Recommender Model
This is the pytorch implementation of our paper at SIGIR 2023:
> [Diffusion Recommender Model](https://arxiv.org/abs/2304.04971)
> 
> Wenjie Wang, Yiyan Xu, Fuli Feng, Xinyu Lin, Xiangnan He, Tat-Seng Chua

## Environment
- Anaconda 3
- python 3.8.10
- pytorch 1.12.0
- numpy 1.22.3

## Usage
### Data
The experimental data are in './datasets' folder, including Amazon-Book, Yelp and MovieLens-1M. Note that the item embedding files of Amazon-book for clean setting and noisy setting are not here due to filesize limits, which are available at [here](https://rec.ustc.edu.cn/share/b6c7dc70-39de-11ef-93a6-a7cb897fc286). Those item embeddings used in L-DiffRec are derived from a pre-trained LightGCN specific to each dataset.

Note that the results on ML-1M differ from those reported in [CODIGEM](https://dl.acm.org/doi/10.1007/978-3-031-10989-8_47), owing to different data processing procedures. CODIGEM did not sort and split the training/testing sets according to timestamps; however, temporal splitting aligns better with the real-world testing.

### Training
To reproduce the results and perform fine-tuning of the hyperparameters, please refer to the model name specified in the **inference.py** file. Ensure that the hyperparameter 'noise_min' is set to a value lower than 'noise_max'.
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

### Inference

1. Download the checkpoints released by us from [here](https://rec.ustc.edu.cn/share/8eac9ff0-39de-11ef-96b7-5ff2a5909c69).
2. Put the 'checkpoints' folder into the current folder.
3. Run inference.py
```
python inference.py --dataset=$1 --gpu=$2
```

### Examples

1. Train DiffRec on Amazon-book under clean setting
```
cd ./DiffRec
sh run.sh amazon-book_clean 5e-5 0 400 [1000] 10 x0 5 0.0001 0.0005 0.005 0 1 log 1 0
```
2. Inference L-DiffRec on Yelp under noisy setting
```
cd ./L-DiffRec
python inference.py --dataset=yelp_noisy --gpu=0
```

## Citation  
If you use our code, please kindly cite:

```
@inproceedings{wang2023diffrec,
title = {Diffusion Recommender Model},
author = {Wang, Wenjie and Xu, Yiyan and Feng, Fuli and Lin, Xinyu and He, Xiangnan and Chua, Tat-Seng},
booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {832–841},
publisher = {ACM},
year = {2023}
}
```
