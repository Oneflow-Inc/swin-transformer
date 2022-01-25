
python3 -m oneflow.distributed.launch --nproc_per_node 8 --master_port 12345  \
        main.py --cfg configs/swin_tiny_patch4_window7_224.yaml \
        --batch-size 32 \
        --tag fix_ddp \
        --libai_config_file configs/libai_dist_config.py
        # --data-path /DATA/disk1/ImageNet/extract/
