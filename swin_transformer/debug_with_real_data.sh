export CUDA_VISIBLE_DEVICES=1

/usr/local/cuda-11.2/nsight-systems-2020.4.3/bin/nsys  profile --stats=true -o swin_nsight_220210@torch  \
python3 debug_with_real_data.py --cfg configs/swin_small_patch4_window7_224.yaml \
        --batch-size 32 \
        --data-path /dataset/extract/ \

