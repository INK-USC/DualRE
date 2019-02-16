#!/bin/sh

# CUDA_VISIBLE_DEVICES=7 ./run-once.sh tacred 0.02 0.98 Self-Training test --seed=1

data_name=$1
labeled_ratio=$2
unlabeled_ratio=$3
model_name=$4
info=$5
extra_args=${6:-""}  # extra arguments for model. Example: "--p_dropout=0.5 --alpha=1"

data_ratio="${labeled_ratio}_${unlabeled_ratio}"

# Define paths
model_dir="saved_models/$data_name/$model_name/$data_ratio/$info/$(date +%m-%d_%H-%M)"
model_prefix="./"

data_dir="${model_prefix}dataset/$data_name"
predictor_dir="${model_prefix}$model_dir/predictor"
selector_dir="${model_prefix}$model_dir/selector"

# Define model parameters
selector_model='pointwise'
integrate_method='intersection'

if [ $model_name = 'Self-Training' ]
then
selector_model='none'
integrate_method='p_only'
fi

if [ $model_name = 'RE-Ensemble' ]
then
selector_model='predictor'
fi

if [ $model_name = 'DualRE' ] # Default DualRE use pointwise retrieval module
then
selector_model='pointwise'
fi

if [ $model_name = 'DualRE-Pairwise' ]
then
selector_model='pairwise'
fi

# Set a larger num_epoch for smaller dataset
if [ $labeled_ratio = '0.05' ] || [ $unlabeled_ratio = '0.05' ]
then
    if [ $data_name = 'semeval' ]
    then
        extra_args="$extra_args --num_epoch 70"
    fi
    if [ $data_name = 'tacred' ]
    then
        extra_args="$extra_args --num_epoch 50"
    fi
fi

train_cmd="python train.py --p_dir ${predictor_dir} --s_dir ${selector_dir} --data_dir $data_dir --labeled_ratio $labeled_ratio --unlabeled_ratio $unlabeled_ratio --selector_model $selector_model --integrate_method $integrate_method $extra_args"
echo $train_cmd
$train_cmd