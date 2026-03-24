#!/bin/bash

array1=('cora' 'yelp' 'ModelNet40' 'NTU2012' 'coauthor_cora' 'house-committees' 'citeseer' 'pubmed' 'actor' 'pokec' 'twitch' '20newsW100' 'walmart-trips' 'coauthor_dblp' 'Mushroom' 'trivago')

k=0.5
savef='tmlr_final.csv'
for i in {1..16}; do
    dataset=${array1[i-1]}
    # h=${hid[i-1]}
    FREE_GPUS=(0 1 2 3)
    if [[ "$dataset" == "trivago" ]]; then
        h=512
        python train_sparse.py --method HGNN --dname $dataset --All_num_layers 3 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 2000 --runs 5 --lr 0.005 --perturb_type replace --perturb_prop 0 --seed 1 --mode NeuralF --keep_ratio $k --fname $savef  --cuda ${FREE_GPUS[0]} --theory &
        python train_sparse.py --method HGNN --dname $dataset --All_num_layers 3 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 2000 --runs 5 --lr 0.01 --perturb_type replace --perturb_prop 0 --seed 1 --mode full --keep_ratio $k --cuda ${FREE_GPUS[1]} --fname $savef --theory &
        python train_sparse.py --method HGNN --dname $dataset --All_num_layers 3 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 2000 --runs 5 --lr 0.005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask+_agn --keep_ratio $k --cuda ${FREE_GPUS[2]} --fname $savef  --theory & # EHGNN-C
        python train_sparse.py --method HGNN --dname $dataset --All_num_layers 3 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 2000 --runs 5 --lr 0.005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask --keep_ratio $k --fname $savef  --cuda ${FREE_GPUS[3]} --theory # EHGNN-F
        python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 2000 --runs 5 --lr 0.005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask_cond --keep_ratio $k --fname $savef  --cuda ${FREE_GPUS[0]} & # EHGNN-F (cond) # Blows up memory on large data
        python train_sparse.py --method HGNN --dname $dataset --All_num_layers 3 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 2000 --runs 5 --lr 0.005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask+ --keep_ratio $k --fname $savef  --cuda ${FREE_GPUS[1]} --theory & # EHGNN-C (cond)
        # python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 2000 --runs 5 --lr 0.005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask_cond --keep_ratio 0.5 --cuda 2 --fname memfix --withchunk --chunk_size 0.25
        python train_sparse.py --method HGNN --dname $dataset --All_num_layers 3 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 2000 --runs 5 --lr 0.005 --perturb_type replace --perturb_prop 0 --seed 1 --mode Neural --keep_ratio $k --fname $savef  --cuda ${FREE_GPUS[2]} --theory &

        python train_sparse.py --method HGNN --dname $dataset --All_num_layers 3 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 2000 --runs 5 --lr 0.005 --perturb_type replace --perturb_prop 0 --seed 1 --mode random --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[3]} --theory 
        # python train_sparse.py --method HGNN --dname $dataset --All_num_layers 3 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 2000 --runs 5 --lr 0.005 --perturb_type replace --perturb_prop 0 --seed 1 --mode min_deg --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[0]} --theory &
        python train_sparse.py --method HGNN --dname $dataset --All_num_layers 3 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 2000 --runs 5 --lr 0.005 --perturb_type replace --perturb_prop 0 --seed 1 --mode effresist --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[1]} --theory & #--approxLinv
        python train_sparse.py --method HGNN --dname $dataset --All_num_layers 3 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 2000 --runs 5 --lr 0.005 --perturb_type replace --perturb_prop 0 --seed 1 --mode degdist --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[0]} --theory 

    else 
        h=512
        e=1000
        python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs $e --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode Neural --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[0]} --theory & # EHGNN-C(cond,LR)
        python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs $e --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode NeuralF --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[1]} --theory & # EHGNN-F(cond,LR)

        python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs $e --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode random --theory  --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[2]} &
        # python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs $e --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode min_deg --theory  --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[3]}
        python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs $e --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode effresist --theory --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[3]}  # --approxLinv

        python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs $e --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode full --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[0]} --theory &
        python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs $e --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[2]} --theory & # EHGNN-F
        python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs $e --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask_cond --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[1]} --theory & # EHGNN-F (cond)
        python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs $e --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask+_agn --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[3]} --theory  # EHGNN-C
        python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs $e --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask+ --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[1]} --theory & # EHGNN-C (cond)
        python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs $e --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode degdist --theory  --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[3]}

    fi
    # ED-GNN
    # python train_sparse.py --method EDGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --feature_noise 0.0 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --aggregate mean --restart_alpha 0.0 --lr 0.0005 --wd 0.0 --epochs 500 --runs 5 --perturb_type replace --perturb_prop 0 --seed 1  --mode full
    # python train_sparse.py --method EDGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --feature_noise 0.0 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --aggregate mean --restart_alpha 0.0 --lr 0.0005 --wd 0.0 --epochs 500 --runs 5 --perturb_type replace --perturb_prop 0 --seed 1  --mode learnmask+_agn --withbucket --num_buckets 8
done 

# array1=('ufg_n0.3' 'ufg_n0.4' 'ufg_n0.8' 'ufg_n0.9')
# k=0.5
# savef='ufg_finalv2.csv'
# for i in {1..4}; do
#     dataset=${array1[i-1]}
#     FREE_GPUS=(0 1 2 3)
#      h=512
#     python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 5 --lr 0.0001 --perturb_type replace --perturb_prop 0 --seed 1 --mode Neural --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[0]} --theory & # EHGNN-C(cond,LR)
#     python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode NeuralF --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[1]} --theory & # EHGNN-F(cond,LR)
#     python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[2]} --theory & # EHGNN-F
#     python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask_cond --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[3]} --theory & # EHGNN-F (cond)
#     python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode full --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[1]} --theory &
#     python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 5 --lr 0.0001 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask+_agn --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[2]} --theory & # EHGNN-C
#     python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask+ --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[3]} --theory  # EHGNN-C (cond)
# done 

# array1=('cora' 'yelp' 'ModelNet40' 'NTU2012' 'coauthor_cora' 'house-committees' 'citeseer' 'pubmed' 'actor' 'pokec' 'twitch' '20newsW100' 'walmart-trips' 'coauthor_dblp' 'Mushroom' 'trivago')
# savef='tmlr_ablation.csv'
# k=0.5
# for i in {1..16}; do 
#     dataset=${array1[i-1]}
#     FREE_GPUS=(0 1 2 3)
#     h=512
#     if [[ "$dataset" == "trivago" ]]; then
#         python train_sparse.py --method HGNN --dname $dataset --All_num_layers 3 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 2000 --runs 5 --lr 0.005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask --keep_ratio $k --fname $savef  --cuda ${FREE_GPUS[0]} --theory & # EHGNN-F
#         python train_sparse.py --method HGNN --dname $dataset --All_num_layers 3 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 2000 --runs 5 --lr 0.005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask --keep_ratio $k --fname $savef  --cuda ${FREE_GPUS[1]} --theory --reg l2 # EHGNN-F w/ L2 reg.
#     else
#         python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[0]} --theory & # EHGNN-F
#         python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[1]} --theory --reg l2 # EHGNN-F w/ L2 reg.
#     fi  
# done
# array1=('cora' 'yelp' 'ModelNet40' 'NTU2012' 'coauthor_cora' 'house-committees' 'citeseer' 'pubmed' 'actor' 'pokec' 'twitch' '20newsW100' 'walmart-trips' 'coauthor_dblp' 'Mushroom' 'trivago')
# savef='tmlr_ablation_topk.csv'
# k=0.5
# for i in {1..16}; do 
#     dataset=${array1[i-1]}
#     FREE_GPUS=(0 1 2 3)
#     h=512
#     if [[ "$dataset" == "trivago" ]]; then
#         python train_sparse.py --method HGNN --dname $dataset --All_num_layers 3 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 2000 --runs 5 --lr 0.005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask --keep_ratio $k --fname $savef  --cuda ${FREE_GPUS[0]} --theory & # EHGNN-F
#         python train_sparse.py --method HGNN --dname $dataset --All_num_layers 3 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 2000 --runs 5 --lr 0.005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask --keep_ratio $k --fname $savef  --cuda ${FREE_GPUS[1]} --theory --sampling topk # EHGNN-F w/ deterministic sampling.
#     else
#         python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[0]} --theory & # EHGNN-F
#         python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask --keep_ratio $k --fname $savef --cuda ${FREE_GPUS[1]} --theory --sampling topk # EHGNN-F w/ deterministic sampling.
#     fi  
# done

# Approximate pseudoinverse
# array1=('yelp' 'walmart-trips' 'coauthor_dblp')
# savef='tmlr_spectral.csv'
# k=0.5
# for i in {1..3}; do 
#     dataset=${array1[i-1]}
#     h=512
#     if [[ "$dataset" == "trivago" ]]; then
#         python train_sparse.py --method HGNN --dname $dataset --All_num_layers 3 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 2000 --runs 5 --lr 0.005 --perturb_type replace --perturb_prop 0 --seed 1 --mode effresist --keep_ratio $k --fname $savef  --cuda $i --theory  & # EHGNN-F
#     else
#         python train_sparse.py --method HGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 256 --wd 0.0 --epochs 1000 --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode effresist --keep_ratio $k --fname $savef --cuda $i --theory & # EHGNN-F
#     fi  
# done