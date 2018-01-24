import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from sklearn import linear_model
import numpy as np

from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from random import shuffle
from sklearn.metrics.classification import classification_report

model = Doc2Vec.load('./spamModel.d2v')

fHam = open('cleanedHam','r')
fSpam = open('cleanedSpam','r')

hamArray=[]
spamArray=[]
for line in fHam:
    hamArray.append(line[:-1])
for line in fSpam:
    spamArray.append(line[:-1])
# print(hamArray)
shuffle(hamArray)
shuffle(spamArray)


# print(len(hamArray), len(spamArray))

trainData = []
testData = []
trainClass = []
testClass = []

for i in range(0,int(len(hamArray)*0.7)):
    trainData.append(model.docvecs['ham_'+str(i)])
    trainClass.append(0)

for i in range(int(len(hamArray)*0.7),len(hamArray)):
    testData.append(model.docvecs['ham_'+str(i)])
    testClass.append(0)


for i in range(0,int(len(spamArray)*0.7)):
    trainData.append(model.docvecs['spam_'+str(i)])
    trainClass.append(1)

for i in range(int(len(spamArray)*0.7),len(spamArray)):
    testData.append(model.docvecs['spam_'+str(i)])
    testClass.append(1)

# print (len(trainData),len(trainClass), len(testData), len(testClass))

trainData=np.array(trainData)
trainClass=np.array(trainClass)
testData=np.array(testData)
testClass=np.array(testClass)


# classifier = linear_model.LogisticRegression()
# classifier = linear_model.SGDClassifier(loss="hinge", penalty="l2")
# classifier = KNeighborsClassifier(n_neighbors=3)
# classifier = tree.DecisionTreeClassifier()
# classifier = GaussianNB()
classifier = svm.SVC(kernel="rbf",verbose=2)
classifier.fit(trainData, trainClass)
target_names = ['Ham','Spam']
#
# LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

testPrediction = classifier.predict(testData)
print(classification_report(testClass,testPrediction,target_names= target_names))
