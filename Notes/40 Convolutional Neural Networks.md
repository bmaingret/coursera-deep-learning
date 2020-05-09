# Convolutional Neural Networks

## Week 1 - Computer Vision

### Idea

The large dimension of the input space when using images requires new algorithms.

Convolute the matrix with a specific filter to detect features.

The idea behind CNN is to treat the filter as parameters to learn.

Technically what is usually called convolution in DL is a cross-correlation.

### Padding

(n*n) matrix convoluted with a (f*f) matrix gives a (n-f+1, n-f+1)

Two issues:

* shrinks the image
* throw away information from the edge

Pad the image with p leading to: (n+p*n+p) matrix convoluted with a (f*f) matrix gives a (n+p-f+1, n+p-f+1).  With n=6, f=4, p=1, your end result would be (6,6).

**Valid convolutions**: no padding
**Same convolutions**: padding to keep the same input size for output size (p=(f-1)/2)

Usually f is an odd number:

* issue with padding
* allow for a central pixel

### Strided convolution

Isntead of moving the filter by 1 step, you move it by a specific stride

(n,n) conv (f, f), with padding p and stride s, floor of ( (n+2p-f)/s +1, (n+2p-f)/s +1 )

### Convolutions over volume

Image: heigth*width*#channels (=depth)

Filter has the same number of channels

### Multiple filters

Stack the output of convolution with each other (similarly to adding channels).

### One layer of CNN

1. Convolution between input and filters (similar to W*A)
2. Add bias (similar to + B)
3. Activation function (similar to relu(WA+B)) -> also stacks together the output of the different filters

**Number of parameters:** 10 filters, (3,3,3) -> (3*3*3+1)*10=280

Note that it doesn't not depends on the input feature space dimension

**Notations**
* filter size: <img src="https://render.githubusercontent.com/render/math?math=f^{[l]}">
* padding: <img src="https://render.githubusercontent.com/render/math?math=p^{[l]}">
* stride: <img src="https://render.githubusercontent.com/render/math?math=s^{[l]}">
* inputdimensino: <img src="https://render.githubusercontent.com/render/math?math=n_H^{[l-1]}*n_W^{[l-1]}*n_C^{[l-1]}">
* output dimension: <img src="https://render.githubusercontent.com/render/math?math=n_H^{[l]}*n_W^{[l]}*n_C^{[l]}">
* filter dimension: <img src="https://render.githubusercontent.com/render/math?math=$f^{[l]}*f^{[l]}*n_C^{[l*1]}$">
* activations: <img src="https://render.githubusercontent.com/render/math?math=$a^{[l]} (n_H^{[l]}, n_W^{[l]}, n_C^{[l]})$">
* weights dimension: <img src="https://render.githubusercontent.com/render/math?math=$(f^{[l]}*f^{[l]}*n_C^{[l-1]})*n_C^{[l]}$">
* bias dimension: <img src="https://render.githubusercontent.com/render/math?math=$n_C^{[l]}$">

### Pooling layers (POOL)

* Max pooling: apply a filter taking the max of the values. 
* Average pooling: apply a filter taking the average of the values. 

Hyperparameters:

* filter size
* stride
* usually no padding
Note that there is no parameters to learn during backpropagation.

Can be applied independently on each channel if multiple channels.

### Full CNN overview

Usually one CONV+POOL is considered as 1 layer (especially since POOL has no parameters t be learned)

For the output of the CNN, consolidate the last POOL output into one vector and add fully connected layer (FC)

## Week 2 - Case studies

### Classic networks
#### LeNet - 5

Simple network, we would use MAX pooling instead now and ReLu.

![](note_images/lenet5.png)

[Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)


#### AlexNet

Similar to LeNet but much bigger and use ReLu

![](note_images/alexnet.png)

[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)


#### VGG-16

Simple layers with regular parameters but huge network.

![](note_images/vgg16.png)

[ImageNet Classification with Deep Convolutional Neural Networks](https://arxiv.org/pdf/1409.1556.pdf)

### ResNets

![](note_images/resnet.png)

[ImageNet Classification with Deep Convolutional Neural Networks](https://arxiv.org/pdf/1512.03385.pdf)

### Inception network

1x1 convolution layer allows to decrease or keep the same number of channels.

[Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842.pdf)

Apply multiple layers after one layer and stack them together/

Issue: computation cost -> use 1x1 conv to reduce number of channels

**Inception module**

![](note_images/inception-module.png)

**Inception network**

Multiple inception modules, with intermediate output branch with softmax to ensure that even intermediate layers are learning, and additional max pool layers to adapt layer size.


### Transfer learning

You save the internal layer to disk and remove the last layers. You can chose where to put the line between what you keep and what you will retrain.

Also possible to use it only as a weight initialization.

### Data augmentation

Stay careful not to overfit your data.

Image augmentation:

* PCA color augmentation
* Color shift
* Rotation
* Flip
* Crop

Implementation remarks:

* Have different threads doing the distorsions and training when you cannot store the augmented data
