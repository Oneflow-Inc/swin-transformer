 python3 -m oneflow.distributed.launch \
    --nproc_per_node 8 --master_port 12345 --node_rank 0 --nnodes 1 \
    main_vit_eager_consistent_use_fake_data.py --cfg configs/vit_base_patch16_224.yaml --batch-size 2
