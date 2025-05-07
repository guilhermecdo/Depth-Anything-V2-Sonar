#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

encoder=vitl
pretrained_from=checkpoints/latest.pth
max_depth=20
image_path=reconstruct.txt
save_path=test2.xyz

python3  depth_to_pointcloud.py\
            --encoder $encoder\
            --load-from $pretrained_from\
            --max-depth $max_depth\
            --img-path $image_path\
            --outdir $save_path\ 