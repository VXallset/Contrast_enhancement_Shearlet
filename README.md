# Contrast_enhancement_Shearlet
Medical image contrast enhancement using Shearlet transformation (ST).

This implementation is used for enhancing the contrast of medical images (X-Ray images, more accurately). It takes several .raw images as input, which are collected from X-Ray machines (without any pre-processing), and generates the enhanced images. 

Aparting from the contrast enhancement, other processes, such as gamma-correction and region of interest extraction using Otsu method, are also applied on the input images so as to improve the results. The time cost for enhancing each image is about 1.0 second (on Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz). 

Given an image, the contrast of the it can be enhanced in the following steps:
- Calculate the spectrum of the image using fast Fourier transformation(FFT).
- Calculate the enhanced spectrum of ST of the given shape (which is the same as the image to be enhanced) using ST.
- Multiply the spectrum of the original image and the spectrum of ST.
- Apply inverse fast Fourier transformation (IFFT) on the result of last step.

Note that FFT and IFFT are used to accelerate the Shearlet Transformation, which is named as _Fast Finite Shearlet Transform_ (FFST).  The theoretical basis can be find [here](https://arxiv.org/pdf/1202.1773.pdf).

In order to find the best combanation of parameters, this project is implemented in functional-style.


# Environment
- python 3.6 or higher
- opencv 2

# Original Images and Enhanced Images (ROI Only) 
<img width="400" height="350" src="/img/1_original.jpeg"/><img width="400" height="350" src="/img/1_enhanced.jpg"/>
<img width="400" height="350" src="/img/2_original.jpeg"/><img width="400" height="350" src="/img/2_enhanced.jpg"/>
<img width="400" height="350" src="/img/3_original.jpeg"/><img width="400" height="350" src="/img/3_enhanced.jpg"/>
<img width="400" height="350" src="/img/4_original.jpeg"/><img width="400" height="350" src="/img/4_enhanced.jpg"/>

# For More
Contact me: vxallset@outlook.com
