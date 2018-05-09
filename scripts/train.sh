set -x
set -e
# ./scripts/train.sh 0 18 synthtext
export CUDA_VISIBLE_DEVICES=$1
IMG_PER_GPU=$2
DATAPATH=$3

CHKPT_PATH=${HOME}/models/seglink/seglink_synthtext
TRAIN_DIR=${HOME}/models/seglink/seglink_icdar2015_384
#TRAIN_DIR=${HOME}/temp/no-use/seglink/seglink_icdar2015_384
#CHKPT_PATH=${HOME}/models/ssd-pretrain/seglink

# get the number of gpus
OLD_IFS="$IFS" 
IFS="," 
gpus=($CUDA_VISIBLE_DEVICES) 
IFS="$OLD_IFS"
NUM_GPUS=${#gpus[@]}

# batch_size = num_gpus * IMG_PER_GPU
BATCH_SIZE=`expr $NUM_GPUS \* $IMG_PER_GPU`

python train_seglink.py \
			--train_dir=${TRAIN_DIR} \
			--num_gpus=${NUM_GPUS} \
			--learning_rate=0.0001 \
			--gpu_memory_fraction=-1 \
			--train_image_width=384 \
			--train_image_height=384 \
			--batch_size=${BATCH_SIZE}\
            --dataset_dir=${DATASET_DIR} \
            --dataset_name=${DATASET} \
            --dataset_split_name=train \
            --train_with_ignored=0 \
			--checkpoint_path=${CHKPT_PATH} \
			--using_moving_average=0
			