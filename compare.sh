set -aux

DATA_PATH="/DATA/disk1/ImageNet/extract/"
BATCH_SIZE=16
TOTAL_ITERS=100


python loss_compare.py --batch_size $BATCH_SIZE \
                       --data_path $DATA_PATH \
                       --total_iters $TOTAL_ITERS \
                       --draw