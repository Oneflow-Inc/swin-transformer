 python3 -m oneflow.distributed.launch \
    --nproc_per_node 2 --master_port 12345 --node_rank 0 --nnodes 1 \
    main_swin_eager_consistent_use_fake_data.py \
    --cfg configs/swin_small_patch4_window7_224.yaml \
    --batch-size 2

#  python3 -m oneflow.distributed.launch --nproc_per_node 2 --node_rank 0 --nnodes 1 test.py
