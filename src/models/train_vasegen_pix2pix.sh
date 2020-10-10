set -ex
cd models/pix2pix

size=256

if [ $size == '512' ]; then

python train.py \
--dataroot ../../data/processed/pix2pix_vase_fragments_512 --name pix2pix_vase_fragments \
--model pix2pix --netG resnet_9blocks --direction BtoA --lambda_L1 100 \
--dataset_mode aligned --norm batch --pool_size 0 --preprocess no \
--batch_size 16 --n_epochs 10 --n_epochs_decay 10 \
# --verbose --n_layers_D 3 --no-flip --continue

elif [ $size == '256' ]; then

python train.py \
--dataroot ../../data/processed/pix2pix_vase_fragments_256 --name pix2pix_vase_fragments \
--model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 \
--dataset_mode aligned --norm batch --pool_size 0 \
--batch_size 16 --n_epochs 10 --n_epochs_decay 10 \
# --preprocess no --verbose --n_layers_D 3 --no-flip --continue

else

echo unknown size $size

fi
