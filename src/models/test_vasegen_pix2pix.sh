set -ex
cd models/pix2pix

size=512

if [ $size == '256' ]; then

model=unet_256
batch=16

elif [ $size == '512' ]; then

model=unet_512
batch=8

else

echo unknown size $size

fi

python test.py \
--dataroot ../../data/processed/pix2pix_vase_examples_512 --name pix2pix_vase_fragments_$size \
--model pix2pix --netG $model --direction BtoA --dataset_mode aligned --norm batch \
--eval --preprocess none
