## Training data infomation
Data used to train the model in the repository can be downloaded [here](https://drive.google.com/file/d/1AiUt7TuIUcVLb3M_iM99yGhJTtuhOB_x/view?usp=sharing). Training data is the en-vi language pair of the [OPUS TED2020v1 data](https://opus.nlpl.eu/TED2020-v1.php).   

## Instruction for using your own data
In case you would like to make your own translator (for example for a different language pair), you need to provide a proper training dataset.

To prepare the training data, simply create **two text files** containing pair of sentences (or pair of paragraphs) in your source and target language. Each sentence (or paragraph) must be **on one line**, i.e., it must contains **only one** '\n' at the end of the sentence (or paragraph).  

For example, if you want to make a translator from Vietnamese to English as I have done, you need to create a `vi.txt` file (source language data) and `en.txt` file (target language data). Below are the first three lines in these two files (data taken from [OPUS TED2020v1](https://opus.nlpl.eu/TED2020-v1.php)):  
    
    File vi.txt:  
    [line 1] Cám ơn rất nhiều, Chris.  
    [line 2] Đây thật sự là một vinh hạnh lớn cho tôi khi có cơ hội được đứng trên sân khấu này hai lần; Tôi thật sự rất cảm kích.  
    [line 3] Tôi thực sự bị choáng ngợp bởi hội nghị này, và tôi muốn cám ơn tất cả các bạn vì rất nhiều nhận xét tốt đẹp về những gì tôi đã trình bày đêm hôm trước.  

    File en.txt:
    [line 1] Thank you so much, Chris.  
    [line 2] And it's truly a great honor to have the opportunity to come to this stage twice; I'm extremely grateful.  
    [line 3] I have been blown away by this conference, and I want to thank all of you for the many nice comments about what I had to say the other night.  

Now you can train your model using `train.py`.

----
**Copyright note:** Please follow the requirements of OPUS or any other place you took data from. 

The training data I have used are taken from the OPUS corpus:  
> Website: http://opus.nlpl.eu
> 
> Please cite the following article if you use any part of the corpus in your own work: J. Tiedemann, 2012, Parallel Data, Tools and Interfaces in OPUS. In Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC 2012)
> 
> This dataset contains a crawl of nearly 4000 TED and TED-X transcripts from July 2020. The transcripts have been translated by a global community of volunteers to more than 100 languages. The parallel corpus is available from https://www.ted.com/participate/translate

