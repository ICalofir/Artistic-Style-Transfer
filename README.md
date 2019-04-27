# Artistic-Style-Transfer

# Info
---
My undergraduate thesis project. In this project I implemented style transfer by combining the content of one image with the style of another image. This project contains the implementation of three different papers: [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), [Deep Photo Style Transfer](https://arxiv.org/abs/1703.07511) and [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155).

Tensorflow: 1.7.0

CUDA: 9.0.176

cuDNN: 7.0.5

# Results
---
### A Neural Algorithm of Artistic Style
<p><img src='examples/content/content1.jpg' height='200' />
<img src='examples/content/content2.jpg' height='200' /></p>

<p><img src='examples/style/style1.jpg' height='100' width='100' />
<img src='examples/result/anaoas_c1s1.png' height='200' />
<img src='examples/result/anaoas_c2s1.png' height='200' /></p>

<p><img src='examples/style/style2.jpg' height='100' width='100' />
<img src='examples/result/anaoas_c1s2.png' height='200' />
<img src='examples/result/anaoas_c2s2.png' height='200' /></p>

<p><img src='examples/style/style3.jpg' height='100' width='100' />
<img src='examples/result/anaoas_c1s3.png' height='200' />
<img src='examples/result/anaoas_c2s3.png' height='200' /></p>

<p><img src='examples/style/style4.jpg' height='100' width='100' />
<img src='examples/result/anaoas_c1s4.png' height='200' />
<img src='examples/result/anaoas_c2s4.png' height='200' /></p>

<p><img src='examples/style/style5.jpg' height='100' width='100' />
<img src='examples/result/anaoas_c1s5.png' height='200' />
<img src='examples/result/anaoas_c2s5.png' height='200' /></p>

<p><img src='examples/style/style6.jpg' height='100' width='100' />
<img src='examples/result/anaoas_c1s6.png' height='200' />
<img src='examples/result/anaoas_c2s6.png' height='200' /></p>

<p><img src='examples/style/style7.jpg' height='100' width='100' />
<img src='examples/result/anaoas_c1s7.png' height='200' />
<img src='examples/result/anaoas_c2s7.png' height='200' /></p>

### Deep Photo Style Transfer
<p><img src='examples/content/d_content1.png' height='200' width='200' />
<img src='examples/style/d_style1.png' height='200' width='200' />
<img src='examples/result/dpst_c1s1.png' height='200' width='200' />
<img src='examples/mask/mask_d_content1.png' height='100' width='100' />
<img src='examples/mask/mask_d_style1.png' height='100' width='100' /></p>

<p><img src='examples/content/d_content2.png' height='200' width='200' />
<img src='examples/style/d_style2.png' height='200' width='200' />
<img src='examples/result/dpst_c2s2.png' height='200' width='200' />
<img src='examples/mask/mask_d_content2.png' height='100' width='100' />
<img src='examples/mask/mask_d_style2.png' height='100' width='100' /></p>

<p><img src='examples/content/d_content3.png' height='200' width='200' />
<img src='examples/style/d_style3.png' height='200' width='200' />
<img src='examples/result/dpst_c3s3.png' height='200' width='200' />
<img src='examples/mask/mask_d_content3.png' height='100' width='100' />
<img src='examples/mask/mask_d_style3.png' height='100' width='100' /></p>

<p><img src='examples/content/d_content4.png' height='200' width='200' />
<img src='examples/style/d_style4.png' height='200' width='200' />
<img src='examples/result/dpst_c4s4.png' height='200' width='200' />
<img src='examples/mask/mask_d_content4.png' height='100' width='100' />
<img src='examples/mask/mask_d_style4.png' height='100' width='100' /></p>

<p><img src='examples/content/d_content5.png' height='200' width='200' />
<img src='examples/style/d_style5.png' height='200' width='200' />
<img src='examples/result/dpst_c5s5.png' height='200' width='200' />
<img src='examples/mask/mask_d_content5.png' height='100' width='100' />
<img src='examples/mask/mask_d_style5.png' height='100' width='100' /></p>

<p><img src='examples/content/d_content6.png' height='200' width='200' />
<img src='examples/style/d_style6.png' height='200' width='200' />
<img src='examples/result/dpst_c6s6.png' height='200' width='200' />
<img src='examples/mask/mask_d_content6.png' height='100' width='100' />
<img src='examples/mask/mask_d_style6.png' height='100' width='100' /></p>

<p><img src='examples/content/d_content7.png' height='200' width='200' />
<img src='examples/style/d_style7.png' height='200' width='200' />
<img src='examples/result/dpst_c7s7.png' height='200' width='200' />
<img src='examples/mask/mask_d_content7.png' height='100' width='100' />
<img src='examples/mask/mask_d_style7.png' height='100' width='100' /></p>

### Perceptual Losses for Real-Time Style Transfer and Super-Resolution
<p><img src='examples/style/style1.jpg' height='170' width='170' />
<img src='examples/content/plfrtst_c1s1.jpg' height='170' width='170' />
<img src='examples/result/plfrtst_c1s1_stil.png' height='170' width='170' />
<img src='examples/content/plfrtst_c2s1.jpg' height='170' width='170' />
<img src='examples/result/plfrtst_c2s1_stil.png' height='170' width='170' /></p>

<p><img src='examples/style/style6.jpg' height='170' width='170' />
<img src='examples/content/plfrtst_c1s6.png' height='170' width='170' />
<img src='examples/result/plfrtst_c1s6_stil.png' height='170' width='170' />
<img src='examples/content/plfrtst_c2s6.jpg' height='170' width='170' />
<img src='examples/result/plfrtst_c2s6_stil.png' height='170' width='170' /></p>

<p><img src='examples/style/style7.jpg' height='170' width='170' />
<img src='examples/content/plfrtst_c1s7.png' height='170' width='170' />
<img src='examples/result/plfrtst_c1s7_stil.png' height='170' width='170' />
<img src='examples/content/plfrtst_c2s7.jpg' height='170' width='170' />
<img src='examples/result/plfrtst_c2s7_stil.png' height='170' width='170' /></p>
