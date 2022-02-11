
python3 -m oneflow.distributed.launch --nproc_per_node 8 --master_port 12345  \
        main.py --cfg configs/swin_tiny_patch4_window7_224.yaml \
        --batch-size 16 \
        --tag libai_graph \
        --libai_config_file configs/libai_dist_config.py \
        # --resume output/swin_tiny_patch4_window7_224/libai_graph/model_10
        # --data-path /DATA/disk1/ImageNet/extract/
