from models.pytorch_biggan.datasets import ImageFolder


data_folder = 'data/processed/vase_imgs/met_all/'
pretrained_folder = 'models/pytorch_biggan/pretrained/100k/'

if __name__ == '__main__':
    vase_img_dataset = ImageFolder(data_folder)

