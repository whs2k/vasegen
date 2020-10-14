# VaseGen

VaseGen uses generative adversarial networks to reconstruct ancient vases! This project started by scraping a bunch of vases from the Met Museum website and aggregating all of their relevant metadata. An example vase is shown here:

![Met Vase Example](https://collectionapi.metmuseum.org/api/collection/v1/iiif/254861/530627/main-image =400x)

These >20,000 vases were then filtered down to exclude fragments and include only terracotta vases. These resulting ~2,500 vases were then digitally fragmented, as shown below, to make a dataset.

![Fragment Example](/examples/frag_example1.jpg)

The fragment dataset was then fit using both BigGAN ([Brock et al.](https://arxiv.org/abs/1809.11096)) and pix2pix ([Zhu et al.](https://arxiv.org/abs/1703.10593)) to create the following results

![VaseGen Example 1](/examples/gen_pix2pix_example1.jpg)
![VaseGen Example 2](/examples/gen_pix2pix_example2.jpg)
![VaseGen Example 3](/examples/gen_pix2pix_example3.jpg)
