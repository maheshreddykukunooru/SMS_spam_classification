# SMS_spam_classification

### Modules needed before running the program
1. [Numpy, scipy, sklearn](https://shanshanchen.com/2013/05/29/install-numpy-scipy-scikit-learn-on-mac-os-x-for-data-miners/)
2. [Tensorflow](https://www.tensorflow.org/install/)
3. Genism  

    
    **pip install --upgrade gensim**
    

### How to run the program

    python cleanData.py  - separate the spam and non-spam data into 2 files
    python train.py      - train the model on the existing data (creates d2v file for storing the model)
    python test.py       - test the accuracy of the model



This repository is my basic step towards **Doc2Vec** a module of Tensorflow. In naive machine learning terms it generates a vector 
from a document. It is related to **word2vec** in many ways. We normally generate this vector using bag of words method where we analyze 
the words present in all documents and using n-grams. But tensorflow provides us with a module Doc2Vec that does this part for us and 
outputs a vector. Basics of Doc2Vec can be found [here](https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e). Even 
the relation between word2vec and doc2vec and how the later is build on top of the other are discussed in that link. 

We have training data for the sms spam data in the *SMSSpamCollection* file from which the spam and non-spam(called ham here) are separated out into two different files (This will be useful while using Doc2Vec) using cleanData.py.

Then train.py file generates vectors for each document (here each message is a document) using Doc2Vec module. LabeledLineSentence of Genism is used to generate the vectors.

### Classification Report
                    precision    recall  f1-score      support

        Ham            0.95      1.00      0.97         1449
        Spam           0.97      0.68      0.80         225

    avg / total        0.95      0.95      0.95         1674
