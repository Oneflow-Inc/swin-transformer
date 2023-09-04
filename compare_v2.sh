set -aux

DATA_PATH="/DATA/disk1/ImageNet/extract/"
BATCH_SIZE=8
TOTAL_ITERS=500


python3 loss_compare_other_nets.py --batch_size $BATCH_SIZE \
                       --data_path $DATA_PATH \
                       --total_iters $TOTAL_ITERS \
                       --draw