export CUDA_VISIBLE_DEVICES=1
python3 debug_with_real_data.py --cfg configs/swin_small_patch4_window7_224.yaml \
        --batch-size 32 \
        --data-path /DATA/disk1/ImageNet/extract/  \
        # --cpu_only \


# with clip_grad
# master 9945 87s  10005 84  9933  88s
# view  11511 84s  11499 90  11499 90s

# without clip_grad
# master 9929 73s  9957  72s 9945  72s
# view  11647 72s  11647 74s 11501 72s