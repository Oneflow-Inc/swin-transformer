
python3 -m oneflow.distributed.launch --nproc_per_node 8 --master_port 12345  \
        main.py --cfg configs/swin_tiny_patch4_window7_224.yaml \
        --batch-size 128 \
        --tag fix_eager_global \
        --data-path /data/ImageNet/extract/
