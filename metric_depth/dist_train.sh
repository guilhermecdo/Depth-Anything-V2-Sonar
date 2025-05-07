#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

epoch=150
bs=7
gpus=1
lr=0.000005
encoder=vitb
dataset=see  #vkitti
img_size=518
min_depth=0
max_depth=20 # 80 for virtual kitti
#pretrained_from=../checkpoints/depth_anything_v2_${encoder}.pth
save_path=/media/guilherme/SSD/VITB-2 # exp/vkitti

#mkdir -p $save_path

python3 -m torch.distributed.launch \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20596 \
    train.py --epoch $epoch --encoder $encoder --bs $bs --lr $lr --save-path $save_path --dataset $dataset \
    --img-size $img_size --min-depth $min_depth --max-depth $max_depth \
    --port 20596 2>&1 | tee -a $save_path/$now.log
