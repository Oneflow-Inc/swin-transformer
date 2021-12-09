python3 -m oneflow.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --cfg configs/swin_small_patch4_window7_224.yaml \
                                                                                          --data-path /DATA/disk1/ImageNet/extract/ \
                                                                                          --batch-size 64 \
                                                                                          --tag swin_small