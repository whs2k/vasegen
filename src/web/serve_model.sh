set -ex

size=512
model=unet_512
batch=8

python -m src.web.flaskapp
# these options were rolled into the .py
# --dataroot ../../data/processed/pix2pix_vase_examples_512 \
# --name pix2pix_vase_fragments_$size \
# --model pix2pix --netG $model --direction BtoA --dataset_mode aligned --norm batch \
# --eval --preprocess none
