''' 
A SIMPLE NEURAL MACHINE TRANSLATOR
Last main update: June 2021
[github link]
quangtn@hcmute.edu.vn

INSTRUCTIONS:
    + Run entire code: if you want to train your model from scratch.
    + Run entire code with load_word_lists=True, load_processed_data=True: if you want to train a new model but already run Part 2 (data are preprocessed). This can save data processing time.
    + Run only Part 1 & Part 4: if you already trained (and saved) a model and want to do prediction (translation).
For other instructions, such as how to prepare your data, please see the github repository given above.

The code below have been successfully run on a system with:
Package         version
------------------------        
python          3.7.10
tensorflow      2.3.0
mosestokenizer  1.1.0
joblib          1.0.1
numpy           1.18.5

'''


# In[1]: PART 1. IMPORT AND FUNCTIONS
#region
import sys
from tensorflow import keras
assert sys.version_info >= (3, 5) # Python ≥3.5 is required
import tensorflow as tf
assert tf.__version__ >= "2.0" # TensorFlow ≥2.0 is required
import numpy as np
import joblib
from mosestokenizer import MosesTokenizer, MosesDetokenizer
import gdown
import zipfile
use_GPU = True 
if use_GPU:
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

# Declarations and functions for data preprocessing (used in Part 2 & Part 4)
eos_id = 0 # end-of-seq token id 
sos_id = 1 # start-of-seq token id
oov_id = 2 # out-of-vocab word id
def word_to_id(word, vocab_list):
    if word in vocab_list:
        return vocab_list.index(word) 
    else:
        return oov_id 
def id_to_word(id, vocab_list):
    if id < len(vocab_list):
        return vocab_list[id]
    else:
        return '<unknown word>' # out-of-vocab (oov) word
#endregion


# In[2]: PART 2. LOAD AND PREPROCESS DATA
# Hyperparameters:
N_WORDS_KEPT = 200 # NOTE: HYPERPARAM. Number of words to keep in each sample (a line in txt files)             
min_occurrences = 4 # NOTE: HYPERPARAM. Each word appear many times in the dataset. We only keep the words that occur > min_occurrences in the dataset. Amitabha 
#region
# LOAD DATA:
# Orginal data source: https://opus.nlpl.eu/TED2020-v1.php
unzip_to_path = r'datasets/TED2020_en-vi_txt'
source_language_data_path = r'datasets/TED2020_en-vi_txt/TED2020.en-vi.vi'
target_language_data_path = r'datasets/TED2020_en-vi_txt/TED2020.en-vi.en'

new_download = True
if new_download:
    url_data = 'https://drive.google.com/u/0/uc?id=1AiUt7TuIUcVLb3M_iM99yGhJTtuhOB_x'
    download_output = 'temp.zip'
    gdown.download(url_data, download_output, quiet=False)
    with zipfile.ZipFile(download_output, 'r') as zip_f:
        zip_f.extractall(unzip_to_path)

f = open(source_language_data_path, 'r', encoding='utf-8')
vi_list = f.readlines()
f.close()
print('VI samples:')
[print(sentence) for sentence in vi_list[:3]]
print('vi_list length:',len(vi_list))

f = open(target_language_data_path, 'r', encoding='utf-8')
en_list = f.readlines()
f.close()
print('\n\nEN samples:')
[print(sentence) for sentence in en_list[:3]]
print('en_list length:',len(en_list))


# PREPROCESS DATA:
# Delete all \n:
# INFO: Mosses tokenizer (used below) reserves punctuation (what we want).
#       but its can NOT deal with \n
vi_list = [i.replace('\n',' ') for i in vi_list]
en_list = [i.replace('\n',' ') for i in en_list]

# Add spaces around digits: (otherwise a lot of numbers, e.g., 102, 103, 25648, are in vocab)
marks = ['0','1','2','3','4','5','6','7','8','9',]  
for mark in marks:
    vi_list = [i.replace(mark,' '+mark+' ') for i in vi_list]
    en_list = [i.replace(mark,' '+mark+' ') for i in en_list]

# Add spaces around punctuation to help MosesTokenizer not to keep these 'words': ----', '---biến', '---bởi', '---chúng', '---và', '-bằng', '-chúng', '-chấp', '-cách', '-có'
marks = ['.', ',', ':', '!', '?', '-', '_']  
for mark in marks:
    vi_list = [i.replace(mark,' '+mark+' ') for i in vi_list]
    en_list = [i.replace(mark,' '+mark+' ') for i in en_list]

# Tokenize text using Mosses tokenizer:
# NOTE: Why choose Mosses tokenizer? See "How Much Does Tokenization Affect Neural Machine Translation?"
load_word_lists = False
if not load_word_lists:
    from mosestokenizer import MosesTokenizer, MosesDetokenizer
    en_tokenize = MosesTokenizer('en')
    vi_tokenize = MosesTokenizer('vi')
    en_list_tokenized = []
    vi_list_tokenized = []
    vi_list_filtered = []
    en_list_filtered = []
    for vi_i, en_i in zip(vi_list, en_list): 
        en_tokens = en_tokenize(en_i) 
        vi_tokens = vi_tokenize(vi_i) 
        if en_tokens!=[] and vi_tokens!=[]: # since some sentences become empty after tokenization
            #!! Truncate sentences !!
            # NOTE: this is NOT encouraged (can strongly affect the performance), but done here due to my weak hardware.
            en_list_tokenized.append(en_tokens[:N_WORDS_KEPT])
            vi_list_tokenized.append(vi_tokens[:N_WORDS_KEPT])                      
            vi_list_filtered.append(vi_i)
            en_list_filtered.append(en_i) 
    en_tokenize.close()
    vi_tokenize.close()

    n_samples = len(vi_list_filtered)
    joblib.dump(n_samples, r'./datasets/n_samples.joblib')
    joblib.dump(en_list_tokenized, r'./datasets/en_list_tokenized.joblib')
    joblib.dump(vi_list_tokenized, r'./datasets/vi_list_tokenized.joblib')
    joblib.dump(en_list_filtered, r'./datasets/en_list_filtered.joblib')
    joblib.dump(vi_list_filtered, r'./datasets/vi_list_filtered.joblib')
    print('Done making word lists.')
else:
    n_samples = joblib.load(r'./datasets/n_samples.joblib')
    en_list_tokenized = joblib.load(r'./datasets/en_list_tokenized.joblib')
    vi_list_tokenized = joblib.load(r'./datasets/vi_list_tokenized.joblib')
    en_list_filtered = joblib.load(r'./datasets/en_list_filtered.joblib')
    vi_list_filtered = joblib.load(r'./datasets/vi_list_filtered.joblib')
    print('Done loading word lists.')

load_processed_data = False
if not load_processed_data:
    # Create vocabularies:
    vi_words = [words for sentence in vi_list_tokenized for words in sentence]
    vi_vocab, vi_counts = np.unique(vi_words, return_counts=True)
    vi_vocab_count = {word:count for word, count in zip(vi_vocab, vi_counts)}
    print(vi_vocab.shape)

    en_words = [words for sentence in en_list_tokenized for words in sentence]
    en_vocab, en_counts = np.unique(en_words, return_counts=True)
    en_vocab_count = {word:count for word, count in zip(en_vocab, en_counts)}
    print(en_vocab.shape)

    # Truncate the vocabulary (keep only words that appear at least min_occurrences times)
    truncated_en_vocab = dict(filter(lambda ele: ele[1]>min_occurrences,en_vocab_count.items()))
    truncated_en_vocab = dict(sorted(truncated_en_vocab.items(), key=lambda item: item[1], reverse=True)) # Just to have low ids for most appeared words
    en_vocab_size = len(truncated_en_vocab)
    print(en_vocab_size)
    joblib.dump(en_vocab_size,r'./datasets/en_vocab_size.joblib')

    truncated_vi_vocab = dict(filter(lambda ele: ele[1]>min_occurrences,vi_vocab_count.items()))
    truncated_vi_vocab = dict(sorted(truncated_vi_vocab.items(), key=lambda item: item[1], reverse=True)) # Just to have low ids for most appeared words
    vi_vocab_size = len(truncated_vi_vocab)
    print(vi_vocab_size)
    joblib.dump(vi_vocab_size,r'./datasets/vi_vocab_size.joblib')

    # Convert words to ids:
    # NOTE: preserve 0, 1, 3 for end-of-seq, start-of-seq, and oov-word token
    vi_vocab_list = ['<eos>', '<sos>', '<oov>']
    vi_vocab_list.extend(list(truncated_vi_vocab.keys()))
    joblib.dump(vi_vocab_list,r'./datasets/vi_vocab_list.joblib')
    en_vocab_list = ['<eos>', '<sos>', '<oov>']
    en_vocab_list.extend(list(truncated_en_vocab.keys()))
    joblib.dump(en_vocab_list,r'./datasets/en_vocab_list.joblib')
 
    # Try encode, decoding some samples:
    temp_vi_encode = [list(map(lambda word: word_to_id(word, vi_vocab_list), sentence)) for sentence in vi_list_tokenized[:2]]
    print('\n',temp_vi_encode)
    temp_vi_decode = [list(map(lambda id: id_to_word(id, vi_vocab_list), sentence)) for sentence in temp_vi_encode]
    print('\n',temp_vi_decode)

    # Convert the whole dataset:
    #   X_vi_data: list of lists of token ids of vi_list_tokenized
    #   Y_en_data: list of lists of token ids for en_list_tokenized
    X_vi_data = [list(map(lambda word: word_to_id(word, vi_vocab_list), sentence)) for sentence in vi_list_tokenized]
    Y_en_data = [list(map(lambda word: word_to_id(word, en_vocab_list), sentence)) for sentence in en_list_tokenized]
    
    # Add end-of-seq and start-of-seq tokens:
    X_vi_data =[[sos_id]+sentence+[eos_id] for sentence in X_vi_data]
    Y_en_data =[[sos_id]+sentence+[eos_id] for sentence in Y_en_data]
    Y_seq_lens = [len(sentence) for sentence in Y_en_data]

    # Pad zero to have all sentences of the same length (required when converting to np.array):
    max_X_len = np.max([len(sentence) for sentence in X_vi_data])
    max_Y_len = np.max([len(sentence) for sentence in Y_en_data])
    X_padded = [sentence + [0]*(max_X_len - len(sentence)) for sentence in X_vi_data]  
    Y_padded = [sentence + [0]*(max_Y_len - len(sentence)) for sentence in Y_en_data]          
    print('Done encoding data.')

    joblib.dump(X_padded, r'./datasets/X_padded.joblib')
    joblib.dump(Y_padded, r'./datasets/Y_padded.joblib')
    joblib.dump(Y_seq_lens, r'./datasets/Y_seq_lens.joblib')
    print('Done saving.')   
else:
    Y_seq_lens = joblib.load(r'./datasets/Y_seq_lens.joblib')
    vi_vocab_size = joblib.load(r'./datasets/vi_vocab_size.joblib')
    en_vocab_size = joblib.load(r'./datasets/en_vocab_size.joblib')
    X_padded = joblib.load(r'./datasets/X_padded.joblib')
    Y_padded = joblib.load(r'./datasets/Y_padded.joblib')
    print('Done loading converted data.')

vocab_X_size = vi_vocab_size + 3
vocab_Y_size = en_vocab_size + 3    
print('\nDONE loading and preprocessing data.')

#endregion


# In[3]: TRAIN AN ENCODER–DECODER MODEL
#region
# Hyperparameters:
n_units = 256 # NOTE: HYPERPARAM. Number of units in each layer. For simplicity, I have set the same number of units for all layers. However, feel free to change this if you wish (you can do that by finding where the variable n_units are in the code and change it one by one). 
embed_X_size = 30 # NOTE: HYPERPARAM. Size of embedding output for encoder.
embed_Y_size = 50 # NOTE: HYPERPARAM. Size of embedding output for decoder.
n_epochs = 100 # NOTE: HYPERPARAM. Number of epochs to run training.
batch_size = 64 # NOTE: HYPERPARAM. batch_size
    
# Encoder (4 GRU layers):
encoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32, name='encoder_inputs')
encoder_embedder = keras.layers.Embedding(vocab_X_size, embed_X_size, mask_zero=True) # set mask_zero=True may cause no GPU training  
encoder_embeddings = encoder_embedder(encoder_inputs)
temp_output = keras.layers.GRU(units=n_units, return_sequences=True)(encoder_embeddings)  
temp_output = keras.layers.GRU(units=n_units, return_sequences=True)(temp_output)  
temp_output = keras.layers.GRU(units=n_units, return_sequences=True)(temp_output)  
encoder_output = keras.layers.GRU(units=n_units, return_state=True, name='encoder_final_layer')(temp_output)  
encoder_outputs, encoder_state = encoder_output

# Decoder (4 GRU layers):
decoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
decoder_embedder = keras.layers.Embedding(input_dim=vocab_Y_size, output_dim=embed_Y_size, name='decoder_embedder', mask_zero=True)
decoder_embeddings = decoder_embedder(decoder_inputs)
temp_output = keras.layers.GRU(units=n_units, return_sequences=True)(decoder_embeddings, initial_state=encoder_state)
temp_output = keras.layers.GRU(units=n_units, return_sequences=True)(temp_output, initial_state=encoder_state)
temp_output = keras.layers.GRU(units=n_units, return_sequences=True)(temp_output, initial_state=encoder_state)
RNN_output = keras.layers.GRU(units=n_units, return_sequences=True)(temp_output, initial_state=encoder_state)
output_layer = keras.layers.Dense(vocab_Y_size, activation='softmax', name='output_layer')   
decoder_output = output_layer(RNN_output)

# The model:
model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[decoder_output])
model.compile(loss="sparse_categorical_crossentropy", optimizer='nadam', metrics=["accuracy"])
model.summary()

#%% Train model:
X = np.array(X_padded) 
Y = np.array(Y_padded)
X_for_decoder = np.c_[np.zeros((n_samples, 1)), Y[:, :-1]] # X_for_decoder = [<pad> Y_t-1]. 0: <padding> or <eos>

new_training = True
if new_training:
    checkpoint_name = r'models/encoder_decoder_multilayerGRU'+'_epoch{epoch:02d}_accuracy{accuracy:.4f}'
    model_checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, monitor='accuracy',save_best_only=True)
    early_stop = keras.callbacks.EarlyStopping(monitor='accuracy',patience=10,restore_best_weights=True)
    tensorboard = keras.callbacks.TensorBoard(r'logs/translator_train_log',embeddings_freq=1, embeddings_metadata='embed_file')
    model.fit([X, X_for_decoder], Y, epochs=n_epochs, batch_size=batch_size,
        callbacks = [model_checkpoint, early_stop, tensorboard] )
    #model.save(r'models/encoder_decoder_multilayerGRU')
    print('DONE training.')
#endregion


# In[4]: MAKE PREDICTION (TRANSLATION)
##### NOTE: specify correct model file name below: #####
model_name = r'models/encoder_decoder_multilayerGRU'
model = keras.models.load_model(model_name) 
#region
def predict_output(X_batch, max_output_length=50):
    n_samples, n_steps = X_batch.shape
    Y_pred = np.ones((n_samples,1), dtype=np.int32) # y_t=0 = <pad> token id
    for i in range(max_output_length):
        Y_probas_next = model.predict([X_batch, Y_pred])[:, i:i+1]
        Y_pred_next = np.argmax(Y_probas_next, axis=-1).astype(dtype=np.int32)
        Y_pred = np.concatenate([Y_pred, Y_pred_next], axis=1)
    # Keep only 1 <eos> token:
    Y_pred_list = []
    for i in range(len(Y_pred)):
        Y_pred_list.append(Y_pred[i])
        Y_pred_list[i] = list(np.trim_zeros(Y_pred_list[i], trim='b'))
        Y_pred_list[i].append(0)
    return Y_pred_list
 
#%% Preprocess test data:
vi_raw_strings = ["Đây là dữ liệu thử nghiệm.", 'Nguyện cầu dịch bệnh sớm kết thúc, thế giới trở lại an bình.', 'Cảm ơn bạn.', ]
print('\n\nInput strings (Vietnamese):')
for seq in vi_raw_strings:
    print('\n    ',seq)
    
# Translate:
vi_vocab_list = joblib.load(r'datasets/vi_vocab_list.joblib')
en_vocab_list = joblib.load(r'datasets/en_vocab_list.joblib')  
def process_data(vi_list):
    # Delete all \n:
    # INFO: Mosses tokenizer (used below) preserves punctuation (what we want).
    #       but its can NOT deal with \n    
    vi_list = [i.replace('\n',' ') for i in vi_list]

    # Add spaces around digits: (otherwise a lot of numbers are in vocab)
    marks = ['0','1','2','3','4','5','6','7','8','9',]  
    for mark in marks:
        vi_list = [i.replace(mark,' '+mark+' ') for i in vi_list]

    # Add spaces around punctuation to help MosesTokenizer not to keep these 'words': ----', '---biến', '---bởi', '---chúng', '---và', '-bằng', '-chúng', '-chấp', '-cách', '-có'
    marks = ['.', ',', ':', '!', '?', '-', '_']  
    for mark in marks:
        vi_list = [i.replace(mark,' '+mark+' ') for i in vi_list]

    # Tokenize text using Mosses tokenizer:
    vi_tokenize = MosesTokenizer('vi')
    vi_list_tokenized = [vi_tokenize(vi_i) for vi_i in vi_list]                      
    vi_tokenize.close()   

    # Convert words to ids:
    X_vi_data = [list(map(lambda word: word_to_id(word, vi_vocab_list), sentence)) for sentence in vi_list_tokenized]

    # Add end-of-seq and start-of-seq tokens:
    X_vi_data =[[sos_id]+sentence+[eos_id] for sentence in X_vi_data]

    # Pad zero to have all sentences of the same length (required when converting to np.array):
    max_X_len = np.max([len(sentence) for sentence in X_vi_data])
    X_padded = [sentence + [0]*(max_X_len - len(sentence)) for sentence in X_vi_data]  
    print('Done encoding data.')

    X_processed = np.array(X_padded)
    return X_processed
X_test = process_data(vi_raw_strings)
Y_pred = predict_output(X_test, max_output_length=25)

# Decode ids to words:
def decode_Y_pred(Y_pred):
    word_seq_list = [list(map(lambda id: id_to_word(id, en_vocab_list), id_seq)) for id_seq in Y_pred]
    word_seq_no_se_tokens = []
    for seq in word_seq_list: # remove <sos>, <eos> tokens
        no_se_seq = [i for i in seq if i!='<sos>' and i!='<eos>'] 
        word_seq_no_se_tokens.append(no_se_seq)
    detokenize = MosesDetokenizer('en')    
    decode_seq = [detokenize(seq) for seq in word_seq_no_se_tokens]
    detokenize.close()
    return decode_seq
print('\n\nTranslation (English):')
for seq in decode_Y_pred(Y_pred):
    print('\n    ',seq)
 
#endregion

 



