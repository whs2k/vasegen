# VaseGen

VaseGen uses generative adversarial networks to reconstruct ancient vases! This project started by scraping a bunch of vases from the Met Museum website and aggregating all of their relevant metadata. An example vase is shown here:

<p align="center">
<img src="https://collectionapi.metmuseum.org/api/collection/v1/iiif/254861/530627/main-image" alt="drawing" alt="Met Vase Example"/>
</p>
<p align="left"/>

These >20,000 vases were then filtered down to exclude fragments and include only terracotta vases. These resulting ~2,500 vases were then digitally fragmented, as shown below, to make a dataset.

<p align="center">
<img src="/examples/frag_example1.jpg" width="800" alt="Fragment Example"/>
</p>
<p align="left"/>

The fragment dataset was then fit using both BigGAN ([Brock et al.](https://arxiv.org/abs/1809.11096)) and pix2pix ([Zhu et al.](https://arxiv.org/abs/1703.10593)) to create the following results

<p align="center">
<img src="/examples/gen_pix2pix_example1.jpg" width="800" alt="VaseGen Example 1"/>
<img src="/examples/gen_pix2pix_example2.jpg" width="800" alt="VaseGen Example 2"/>
<img src="/examples/gen_pix2pix_example3.jpg" width="800" alt="VaseGen Example 3"/>
</p>
<p align="left"/>
