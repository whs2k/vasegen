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

python train.py \
--dataroot ../../data/processed/pix2pix_vase_fragments_512 --name pix2pix_vase_fragments_$size \
--model pix2pix --netG $model --direction BtoA --lambda_L1 100 \
--dataset_mode aligned --norm batch --pool_size 0 \
--batch_size $batch --n_epochs 20 --n_epochs_decay 5 --preprocess none
# --num_threads 4 --preprocess resize_and_crop --save_epoch_freq 1000
# --verbose --n_layers_D 3 --no-flip --continue
