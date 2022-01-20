export CUDA_VISIBLE_DEVICES=1

/usr/local/cuda-11.2/nsight-systems-2020.4.3/bin/nsys  profile --stats=true -o swin_nsight_220119@view \
python3 debug_with_real_data.py --cfg configs/swin_small_patch4_window7_224.yaml \
        --batch-size 32 \
        --data-path /dataset/extract/ \
        # --cpu_only \


# with clip_grad
# master 9945 87s  10005 84  9933  88s
# view  11511 84s  11499 90  11499 90s

# without clip_grad
# master 9929 73s  9957  72s 9945  72s
# view  11647 72s  11647 74s 11501 72s



# master@0.6.0.dev20220105+cu112 VS view@0.6.0+cu112.git.0ce291bbd
# with clip_grad
# master 9030  64s   9062  65s   9314  64s
# view   9818  65s   10650 69s   9916  68s

# without clip_grad
# master 9068  58s   9030  54s   9048  56s
# view   9816  60s   9864  60s   9818  61s   

