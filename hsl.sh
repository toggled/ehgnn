#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# array1=('cora' 'yelp' 'ModelNet40' 'NTU2012' 'coauthor_cora' 'house-committees' 'citeseer' 'pubmed' 'actor' 'pokec' 'twitch' '20newsW100' 'walmart-trips' 'coauthor_dblp' 'Mushroom' 'trivago')
# array1=('cora' 'yelp' 'ModelNet40' 'NTU2012' 'coauthor_cora' 'house-committees' 'citeseer' 'pubmed' 'actor' 'pokec' 'trivago')
# hid=(32 16 16 16 8 8 16 31 16 32 16 8 32 16 1024)
array1=('20newsW100')
# for dataset in "${array1[@]}"; do
k=0.5
# savef='hslv2.csv'
# savef='hsl_cont0_5.csv'
savef='tmlr_final.csv'
for i in {1..1}; do
    dataset=${array1[i-1]}
    # h=${hid[i-1]}
    FREE_GPUS=(0 1 2 3)
    # while [ ${#FREE_GPUS[@]} -le 2 ]; do
    #     for j in {0..3}; do
    #         if ! nvidia-smi -i $j | grep -qi "python"; then
    #             FREE_GPUS+=($j)
    #         fi

    #         # stop once we found 2
    #         if [ ${#FREE_GPUS[@]} -eq 2 ]; then
    #             break
    #         fi
    #     done
    #     sleep 1
    # done
    # echo "Using GPUs: ${FREE_GPUS[@]} for dataset $dataset"
    if [[ "$dataset" == "trivago" ]]; then
        h=128
        python train_sparse.py --dname $dataset --method HSL --mode full --patience 100 --epochs 1000 --lr 0.0005 --dropout 0.5 --hsl_tau 0.5 --hsl_contrastive_weight 0.1 --hsl_lambda 1.0 --All_num_layers 1 --MLP_hidden $h --Classifier_hidden 64 --seed 1 --runs 3 --display_step 50 --keep_ratio $k --fname $savef --cuda 3 --reg l2 & # ${FREE_GPUS[$(( i % 4 ))]}
    else 
        h=128
        # For yelp
        if [[ "$dataset" == "yelp" ]]; then
            h=64
            python train_sparse.py --dname $dataset --method HSL --mode full --patience 100 --wd 0 --epochs 1000 --lr 0.0005 --dropout 0.5 --hsl_tau 0.5 --hsl_contrastive_weight 0.1 --hsl_lambda 0.7 --All_num_layers 1 --MLP_hidden $h --Classifier_hidden 64 --seed 1 --runs 3 --display_step 50 --keep_ratio $k --fname $savef --cuda 1 --reg l2 
        elif [[ "$dataset" == "coauthor_dblp" ]]; then
            # python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 500 --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode full --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[1]} &
            python train_sparse.py --dname $dataset --method HSL --mode full --patience 100 --wd 5e-4 --epochs 1000 --lr 0.0005 --dropout 0.5 --hsl_tau 0.9 --hsl_contrastive_weight 0.3 --hsl_lambda 0.9 --All_num_layers 1 --MLP_hidden $h --Classifier_hidden 64 --seed 1 --runs 3 --display_step 50 --keep_ratio $k --fname $savef --cuda 1 --reg l2 
        elif [[ "$dataset" == "cora" ]]; then
            h=256
            python train_sparse.py --dname $dataset --method HSL --mode full --patience 100 --wd 5e-4 --epochs 1000 --lr 0.0005 --dropout 0.5 --hsl_tau 0.5 --hsl_contrastive_weight 0.1 --hsl_lambda 0.7 --All_num_layers 1 --MLP_hidden $h --Classifier_hidden 64 --seed 1 --runs 3 --display_step 50 --keep_ratio $k --fname $savef --cuda 1 --reg l2 

        elif [[ "$dataset" == "ModelNet40" ]]; then
            h=128
            python train_sparse.py --dname $dataset --method HSL --mode full --patience 100 --wd 5e-4 --epochs 1000 --lr 0.0005 --dropout 0.5 --hsl_tau 0.9 --hsl_contrastive_weight 0.1 --hsl_lambda 0.7 --All_num_layers 1 --MLP_hidden $h --Classifier_hidden 64 --seed 1 --runs 3 --display_step 50 --keep_ratio $k --fname $savef --cuda 1 --reg l2 
            # python train_sparse.py --dname $dataset --method HSL --mode full --patience 100 --wd 5e-4 --epochs 500 --lr 0.0005 --dropout 0.5 --hsl_tau 0.5 --hsl_contrastive_weight 0.5 --hsl_lambda 0.5 --All_num_layers 1 --MLP_hidden $h --Classifier_hidden 64 --seed 1 --runs 3 --display_step 50 --keep_ratio $k --fname $savef --cuda 1 --reg l2 

        else 
            python train_sparse.py --dname $dataset --method HSL --mode full --patience 100 --wd 5e-4 --epochs 1000 --lr 0.0005 --dropout 0.5 --hsl_tau 0.8 --hsl_contrastive_weight 0.1 --hsl_lambda 0.95 --All_num_layers 1 --MLP_hidden $h --Classifier_hidden 64 --seed 1 --runs 3 --display_step 50 --keep_ratio $k --fname $savef --cuda 0 --reg l2 

        fi 
        
    fi
    sleep 1
done 
