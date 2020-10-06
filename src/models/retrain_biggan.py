from models.pytorch_biggan.datasets import ImageFolder


pretrained_folder = 'models/pytorch_biggan/pretrained/100k/'

if __name__ == '__main__':
    vase_img_dataset = ImageFolder(data_folder)

