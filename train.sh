export CUDA_VISIBLE_DEVICES=0,1,2,4
# export CUDA_VISIBLE_DEVICES=5,6,7
python -m torch.distributed.launch \
    --master_port=29500 \
    --nproc_per_node=4 \
    --use_env tools/train.py \
    --cfg configs/face.yaml