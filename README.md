# Vietnamese to English Translator
***\*\*Note: the model is still under training, new (and better) version will be uploaded.\*\****

## What is this?
This repository contains code for training a simple neural machine translator as well as a trained model for the Vietnamese-to-English translator.
## How to use it
#### *1. Using it as a translator* (like Google Translate, but not that good of course)
Simply go to this [Google Colab](https://colab.research.google.com/drive/1aOFww6iGrD7LoXS0N1AgqTrv0QgsXiRw?usp=sharing), then click **Run cell** (Ctrl-Enter), input a sentence in Vietnamese and click **Translate** (Please be patient since the model is a bit heavy for Colab to handle.).  

![Demo using the translator on Colab](/resources/demo.PNG "Hope you enjoy this!") 

#### *2. Using the training code to create your own translator*
Steps to do:

    1. Prepare you data (see "datasets" for more details)
    2. (Optional) Customize the model in "train.py"
    3. Run "train.py" 

You can easily customize the model by changing hyperparameters put at the beginning of code parts (marked with comments `NOTE: HYPERPARAM`) (see `train.py`).  
Please be aware that the training process may take days to finish, depending on your customized model and your computer.  
After training, you can deploy your model on, for example, a Colab as I have done above.  

## Brief info about the model
The current model (in `train.py`) is a simple encoder-decoder with 4-GRU-layer encoder and decoder. Due to the lack of resources, attention mechanism and bidirectional RNNs have not been used.  
Beam search or randomly translation.

The model is trained with the **OPUS TED2020-v1** en-vi data with more than 300.000 pairs of text sequences. See `dataset` folder for details.  

***Note for ones who want to implement attention mechanism:*** due to some bug in the current Tensorflow (version 2.5.0) and Tensorflow Addons (version 0.13 as I have tried), we cannot implement attention mechanism in eager mode. If you wish to, you have to use the subclassing API. 



