"""
import os
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import pandas as pd
import numpy as np

train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)
test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
y = train["sentiment"]

print ("Cleaning and parsing movie reviews...\n")
traindata = []
for i in range( 0, len(train["review"])):
    traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], False)))
testdata = []
for i in range(0,len(test["review"])):
    testdata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], False)))
print ('vectorizing... ')
tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,stop_words = 'english')

X_all = traindata + testdata
lentrain = len(traindata)

print ("fitting pipeline... ",tfv.fit(X_all))
X_all = tfv.transform(X_all)

X = X_all[:lentrain]
X_test = X_all[lentrain:]

model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None)
print ("20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(model, X, y, cv=20, scoring='roc_auc')))

print ("Retrain on all training data, predicting test labels...\n")
model.fit(X,y)
result = model.predict(X_test)
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=3)
print ("Wrote results to Bag_of_Words_model.csv")
"""


"""
import os
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import pandas as pd
import numpy as np

train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, \
                delimiter="\t", quoting=3)
test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", \
               quoting=3 )
y = train["sentiment"]
print ("Cleaning and parsing movie reviews...\n")
traindata = []
for i in range( 0, len(train["review"])):
    traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], False)))
testdata = []
for i in range(0,len(test["review"])):
    testdata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], False)))
print ('vectorizing... ')
tfv = TfidfVectorizer(min_df=3,  max_features=None,
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')
X_all = traindata + testdata
lentrain = len(traindata)

print ("fitting pipeline... ")
tfv.fit(X_all)
X_all = tfv.transform(X_all)

X = X_all[:lentrain]
X_test = X_all[lentrain:]

model = LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                         C=1, fit_intercept=True, intercept_scaling=1.0,
                         class_weight=None, random_state=None)
print ("20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(model, X, y, cv=20, scoring='roc_auc')))

print ("Retrain on all training data, predicting test labels...\n")
model.fit(X,y)
result = model.predict_proba(X_test)[:,1]
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=3)
print ("Wrote results to Bag_of_Words_model.csv")

"""


"""
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from KaggleWord2VecUtility import KaggleWord2VecUtility

train_file = 'data/labeledTrainData.tsv'
unlabeled_train_file = 'data/unlabeledTrainData.tsv'
test_file = 'data/testData.tsv'
output_file = 'data/submit.csv'

train = pd.read_csv( train_file, header = 0, delimiter = "\t", quoting = 3 )
test = pd.read_csv( test_file, header = 0, delimiter = "\t", quoting = 3 )
unlabeled_train = pd.read_csv( unlabeled_train_file, header = 0, delimiter= "\t", quoting = 3 )

print("Parsing train reviews...")

clean_train_reviews = []
for review in train['review']:
    clean_train_reviews.append( " ".join( KaggleWord2VecUtility.review_to_wordlist( review )))

unlabeled_clean_train_reviews = []
for review in unlabeled_train['review']:
    unlabeled_clean_train_reviews.append( " ".join( KaggleWord2VecUtility.review_to_wordlist( review )))

print("Parsing test reviews...")

clean_test_reviews = []
for review in test['review']:
    clean_test_reviews.append( " ".join( KaggleWord2VecUtility.review_to_wordlist( review )))

print("Vectorizing...")

vectorizer = TfidfVectorizer( min_df=2, max_df=0.95, max_features = 200000, ngram_range = ( 1, 4 ),
                              sublinear_tf = True )

vectorizer = vectorizer.fit(clean_train_reviews + unlabeled_clean_train_reviews)
train_data_features = vectorizer.transform( clean_train_reviews )
test_data_features = vectorizer.transform( clean_test_reviews )

print("Reducing dimension...")

from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif
fselect = SelectKBest(chi2 , k=70000)
train_data_features = fselect.fit_transform(train_data_features, train["sentiment"])
test_data_features = fselect.transform(test_data_features)

print("Training...")

model1 = MultinomialNB(alpha=0.0005)
model1.fit( train_data_features, train["sentiment"] )

model2 = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)
model2.fit( train_data_features, train["sentiment"] )

p1 = model1.predict_proba( test_data_features )[:,1]
p2 = model2.predict_proba( test_data_features )[:,1]

print("Writing results...")

output = pd.DataFrame( data = { "id": test["id"], "sentiment": p2} )
output.to_csv( output_file, index = False, quoting = 3 )

"""

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility

train_file = 'data/labeledTrainData.tsv'
unlabeled_train_file = 'data/unlabeledTrainData.tsv'
test_file = 'data/testData.tsv'
output_file = 'data/submit.csv'

train = pd.read_csv( train_file, header = 0, delimiter = "\t", quoting = 3 )
test = pd.read_csv( test_file, header = 0, delimiter = "\t", quoting = 3 )
unlabeled_train = pd.read_csv( unlabeled_train_file, header = 0, delimiter= "\t", quoting = 3 )

print("Parsing train reviews...")

clean_train_reviews = []
for review in train['review']:
    clean_train_reviews.append( " ".join( KaggleWord2VecUtility.review_to_wordlist( review )))

unlabeled_clean_train_reviews = []
for review in unlabeled_train['review']:
    unlabeled_clean_train_reviews.append( " ".join( KaggleWord2VecUtility.review_to_wordlist( review )))

print("Parsing test reviews...")

clean_test_reviews = []
for review in test['review']:
    clean_test_reviews.append( " ".join( KaggleWord2VecUtility.review_to_wordlist( review )))

print("Vectorizing...")

vectorizer = TfidfVectorizer( min_df=2, max_df=0.95, max_features = 200000, ngram_range = ( 1, 4 ),
                              sublinear_tf = True )

vectorizer = vectorizer.fit(clean_train_reviews + unlabeled_clean_train_reviews)
train_data_features = vectorizer.transform( clean_train_reviews )
test_data_features = vectorizer.transform( clean_test_reviews )

print("Reducing dimension...")

from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif
fselect = SelectKBest(chi2 , k=70000)
train_data_features = fselect.fit_transform(train_data_features, train["sentiment"])
test_data_features = fselect.transform(test_data_features)

print("Training...")

model = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)
model.fit( train_data_features, train["sentiment"] )

p = model.predict_proba( test_data_features )[:,1]

print("Writing results...")

output = pd.DataFrame( data = { "id": test["id"], "sentiment": p} )
output.to_csv( output_file, index = False, quoting = 3 )
