# Vietnamese to English Translator
## What is this?
This repository contains code for training a neural machine translator as well as a trained model for the Vietnamese-to-English translator.
## How to use it
#### *1. Using it as a translator* (like Google Translate, but not that good of course)
Simply go to this [Colab](https://colab.research.google.com/drive/1aOFww6iGrD7LoXS0N1AgqTrv0QgsXiRw?usp=sharing), click Run cell** input a sentence in Vietnamese and click **Translate**).
[de hinh chup demo colab vo day]

#### *2. Using the training code to create your own translator*
Steps to do:

    2.1. Prepare you data
    hk
    ljl
jjl
## Brief info about the model
The current model (in `train.py`) is a simple encoder-decoder with 4-GRU-layer encoder and decoder. Due to the lack of resources, attention mechanism and bidirectional RNNs have not been used.  

The model is trained with the **OPUS TED2020-v1** en-vi data with more than 300.000 pairs of text sequences. See `dataset` folder for details.  

***Note for ones who want to implement attention mechanism:*** due to some bug in the current Tensorflow (version 2.5.0) and Tensorflow Addons (version 0.13 as I have tried), we cannot implement attention mechanism in eager mode. If you wish to, you have to use the subclassing API. 



