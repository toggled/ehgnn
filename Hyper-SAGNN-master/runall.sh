#!/bin/bash
model='ehgnn'
gpus=(0 1 2 3)
for k in 0.5; do
    for dataset in wordnet drug MovieLens; do
        i=1
        for impl in v1; do
            python trainer.py --data $dataset --model baseline --feature adj --device cuda:${gpus[0]} --hypersagnn-impl $impl &
            python trainer.py --data $dataset --model $model --mode random --feature adj --keep-ratio $k --device cuda:${gpus[1]} --hypersagnn-impl $impl &
            python trainer.py --data $dataset --model $model --mode degdist --feature adj --keep-ratio $k --device cuda:${gpus[2]} --hypersagnn-impl $impl & 
            python trainer.py --data $dataset --model $model --mode effresist --feature adj --keep-ratio $k --device cuda:${gpus[$i]} --hypersagnn-impl $impl &
            python trainer.py --data $dataset --model $model --mode learnmask --feature adj --keep-ratio $k --device cuda:${gpus[0]} --hypersagnn-impl $impl &
            python trainer.py --data $dataset --model $model --mode learnmask+ --feature adj --keep-ratio $k --device cuda:${gpus[1]} --hypersagnn-impl $impl &
            python trainer.py --data $dataset --model $model --mode learnmask_cond --feature adj --keep-ratio $k --device cuda:${gpus[2]} --hypersagnn-impl $impl & 
            python trainer.py --data $dataset --model $model --mode learnmask+_agn --feature adj --keep-ratio $k --device cuda:${gpus[3]} --hypersagnn-impl $impl
            python trainer.py --data $dataset --model $model --mode Neural --feature adj --keep-ratio $k --device cuda:${gpus[3]} --hypersagnn-impl $impl
            python trainer.py --data $dataset --model $model --mode NeuralF --feature adj --keep-ratio $k --device cuda:${gpus[0]} --hypersagnn-impl $impl
        done 
    done 
done 