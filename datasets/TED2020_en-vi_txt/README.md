# Instruction
Download the training data [here](google.com). 
The training data is OPUS TED2020v1 data, can also be find here.  

# Instruction to use your own data
Simply prepare **two text files** containing pair of sentences (or pair of paragraphs) in your source and target language. Each sentence (or paragraph) must be **on one line** (i.e., ended with '\n').  

For example, if you want to make a translator from Vietnamese to English, create a "vi.txt" file (source language samples) and "en.txt" file (target language samples). Below are the first three lines in these two files (data from [OPUS TED2020v1](https://opus.nlpl.eu/TED2020-v1.php)):  
    
    File vi.txt:  
    [line 1] Cám ơn rất nhiều, Chris.  
    [line 2] Đây thật sự là một vinh hạnh lớn cho tôi khi có cơ hội được đứng trên sân khấu này hai lần; Tôi thật sự rất cảm kích.  
    [line 3] Tôi thực sự bị choáng ngợp bởi hội nghị này, và tôi muốn cám ơn tất cả các bạn vì rất nhiều nhận xét tốt đẹp về những gì tôi đã trình bày đêm hôm trước.  

    File en.txt:
    [line 1] Thank you so much, Chris.  
    [line 2] And it's truly a great honor to have the opportunity to come to this stage twice; I'm extremely grateful.  
    [line 3] I have been blown away by this conference, and I want to thank all of you for the many nice comments about what I had to say the other night.  


If you want to train a translator for different language pair, find and prepare relavent data as the following steps: 
 1. step 1
 2. sajfas
dsaf 


====
Corpus Name: TED2020
     Package: TED2020.en-vi in Moses format
     Website: http://opus.nlpl.eu/TED2020-v1.php
     Release: v1
Release date: Mon Nov 30 01:54:29 EET 2020
     License: Please respect the <a href=https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy>TED Talks Usage Policy</a>

This corpus is part of OPUS - the open collection of parallel corpora
OPUS Website: http://opus.nlpl.eu

Please cite the following article if you use any part of the corpus in your own work: J. Tiedemann, 2012, Parallel Data, Tools and Interfaces in OPUS. In Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC 2012)

This dataset contains a crawl of nearly 4000 TED and TED-X transcripts from July 2020. The transcripts have been translated by a global community of volunteers to more than 100 languages. The parallel corpus is available from https://www.ted.com/participate/translate

