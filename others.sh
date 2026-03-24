array1=('ModelNet40' 'NTU2012' 'Mushroom' 'coauthor_cora' 'yelp' 'house-committees' 'cora' 'citeseer' 'pubmed' 'actor' 'pokec' 'twitch' '20newsW100' 'walmart-trips' 'coauthor_dblp')
e=2000

savef='otherbackbones.csv'
# for dataset in "${array1[@]}"; do
for i in {1..15}; do
    dataset=${array1[i-1]}
    if [[ "$dataset" == "trivago" ]]; then
        python train_sparse.py --method HyperUFG --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs $e --runs 5 --lr 0.001 --perturb_type replace  --perturb_prop 0 --seed 1 --mode full --reg none --keep_ratio 0.5 --fname $savef --theory --hyperufg_alpha 0.1 --hyperufg_lambda 0.5 --hyperufg_cheb_order 2 --cuda 0 &
        python train_sparse.py --method HyperUFG --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs $e --runs 5 --lr 0.001 --perturb_type replace  --perturb_prop 0 --seed 1 --mode learnmask --reg none --keep_ratio 0.5 --fname $savef --theory --hyperufg_alpha 0.1 --hyperufg_lambda 0.5 --hyperufg_cheb_order 2 --cuda 1 
        python train_sparse.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 1024 --Classifier_hidden 128 --wd 0.0 --epochs $e --runs 5 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask --reg none --keep_ratio 0.5 --fname $savef --theory  --cuda 2 
        python train_sparse.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 1024 --Classifier_hidden 128 --wd 0.0 --epochs $e --runs 5 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed 1 --mode full --reg none --keep_ratio 0.5 --fname $savef --theory --cuda 3

    else
        python train_sparse.py --method HyperUFG --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 128 --wd 0.0 --epochs $e --runs 5 --lr 0.001 --perturb_type replace  --perturb_prop 0 --seed 1 --mode full --reg none --keep_ratio 0.5 --fname $savef --theory --hyperufg_alpha 0.1 --hyperufg_lambda 0.5 --hyperufg_cheb_order 2 --cuda 0 
        python train_sparse.py --method HyperUFG --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 128 --wd 0.0 --epochs $e --runs 5 --lr 0.001 --perturb_type replace  --perturb_prop 0 --seed 1 --mode learnmask --reg none --keep_ratio 0.5 --fname $savef --theory --hyperufg_alpha 0.1 --hyperufg_lambda 0.5 --hyperufg_cheb_order 2 --cuda 1 
        # python train_sparse.py --method CEGAT --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 128 --wd 0.0 --epochs $e --runs 5 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask --reg none --keep_ratio 0.5 --fname $savef --theory &
        # python train_sparse.py --method CEGCN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 128 --wd 0.0 --epochs $e --runs 5 --cuda 1 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask --reg none --keep_ratio 0.5 --fname $savef --theory &
        python train_sparse.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs $e --runs 5 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask --reg none --keep_ratio 0.5 --fname $savef --theory  --cuda 2 
        # python train_sparse.py --method CEGAT --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 128 --wd 0.0 --epochs $e --runs 5 --cuda 0 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed 1 --mode full --reg none --keep_ratio 0.5 --fname $savef --theory &
        # python train_sparse.py --method CEGCN --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 128 --wd 0.0 --epochs $e --runs 5 --cuda 1 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed 1 --mode full --reg none --keep_ratio 0.5 --fname $savef --theory &
        python train_sparse.py --method AllSetTransformer --dname $dataset --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --epochs $e --runs 5 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed 1 --mode full --reg none --keep_ratio 0.5 --fname $savef --theory --cuda 3
    fi
done 


array1=('cora' 'yelp' 'ModelNet40' 'NTU2012' 'coauthor_cora' 'house-committees' 'citeseer' 'pubmed' 'actor' 'pokec' 'twitch' '20newsW100' 'walmart-trips' 'coauthor_dblp' 'trivago' 'Mushroom')
# for dataset in "${array1[@]}"; do
for i in {1..15}; do
    dataset=${array1[i-1]}
    # h=${hid[i-1]}
    FREE_GPUS=(0 1 2 3)
    echo "Using GPUs: ${FREE_GPUS[@]} for dataset $dataset"
    if [[ "$dataset" == "trivago" ]]; then
        h=1024
        python train_sparse.py --method EDGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 128 --aggregate mean --restart_alpha 0.0 --wd 0.0 --epochs 2000 --runs 5 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed 1 --mode full --keep_ratio 0.5 --cuda ${FREE_GPUS[0]}  --fname $savef --theory &
        python train_sparse.py --method EDGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 128 --aggregate mean --restart_alpha 0.0 --wd 0.0 --epochs 2000 --runs 5 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask --keep_ratio 0.5 --cuda ${FREE_GPUS[1]} --fname $savef --theory

    elif [[ "$dataset" == "yelp" ]]; then
        h=256
        # python train_sparse.py --method EDGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 128 --aggregate mean --restart_alpha 0.0 --wd 0.0 --epochs 1000 --runs 5 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed 1 --mode full --keep_ratio 0.5 --cuda ${FREE_GPUS[0]} --fname $savef --theory &
        python train_sparse.py --method EDGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 128 --aggregate mean --restart_alpha 0.0 --wd 0.0 --epochs 1000 --runs 5 --lr 0.001 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask --keep_ratio 0.5 --cuda ${FREE_GPUS[1]} --fname $savef --theory
    else
        h=512
        python train_sparse.py --method EDGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 128 --aggregate mean --restart_alpha 0.0 --wd 0.0 --epochs 1000 --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode full --keep_ratio 0.5 --cuda ${FREE_GPUS[0]} --fname $savef --theory & 
        python train_sparse.py --method EDGNN --dname $dataset --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden $h --Classifier_hidden 128 --aggregate mean --restart_alpha 0.0 --wd 0.0 --epochs 1000 --runs 5 --lr 0.0005 --perturb_type replace --perturb_prop 0 --seed 1 --mode learnmask --keep_ratio 0.5 --cuda ${FREE_GPUS[0]} --fname $savef --theory
    fi
done 