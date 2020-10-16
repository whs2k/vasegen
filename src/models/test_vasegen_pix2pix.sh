set -ex
python test.py --dataroot ../../data/processed/pix2pix_vase_examples_512 --name downstairs --model pix2pix --netG unet_512 --direction BtoA --dataset_mode aligned --norm batch
