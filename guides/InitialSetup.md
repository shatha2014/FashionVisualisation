# Description

This is a guide for setting up the environment required to run the code in https://github.com/jalused/Deconvnet-keras. The steps in this guide consider a clean installation of Centos 6.9.

## Install Anaconda

Download Anaconda

`curl -O https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh`

Run the installer (yes to everything)

`bash Anaconda3-2018.12-Linux-x86_64.sh`

## Install Theano

`pip install theano`

## Install Keras

`pip install keras==1.1`

## Set theano as the backend for keras

In the home directory:

`sed -i 's/tensorflow/theano/g' ./.keras/keras.json`

`sed -i 's/tf/th/g' ./.keras/keras.json`


## Install Argparse

`pip install argparse==1.0`

## Clone the project

`git clone git://github.com/jalused/Deconvnet-keras.git`

## Fix the code for Python 3

Edit the floating point division symbol to the integer division symbol (/ -> //) in lines 219, 220, 254, 255.

Example (line 219):

`out_shape[2] = out_shape[2] / poolsize[0]`

to

`out_shape[2] = out_shape[2] // poolsize[0]`

## Try the code

In the Deconvnet-keras directory:

`python Deconvnet-keras.py husky.jpg`

Afterwards check the result at:

`results/block5_conv3_0_max.png`

(block5_conv3_0_max is the default feature to visualize)