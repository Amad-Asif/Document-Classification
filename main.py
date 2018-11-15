from sklearn.datasets import fetch_20newsgroups

'''
The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned
(nearly) evenly across 20 different newsgroups.
'''

twenty_train = fetch_20newsgroups(data_home='data', subset='train', shuffle=True, download_if_missing=True)

'''
Text files are actually series of words (ordered). In order to run machine learning algorithms we need
to convert the text files into numerical feature vectors. We will be using bag of words model.

We segment each text file into words (for English splitting by space), and count # of times each word
occurs in each document and finally assign each word an integer id. Each unique word in our dictionary
will correspond to a feature (descriptive feature).
'''

'''
Scikit-learn has a high level component which will create feature vectors for us ‘CountVectorizer’.
'''

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print (X_train_counts.shape) # (11314-n_Samples, 130107-n_features)
print (X_train_counts[1])
print ("-----")
print (X_train_counts[2])
'''
Here by doing ‘count_vect.fit_transform(twenty_train.data)’, we are learning the vocabulary dictionary
and it returns a Document-Term matrix. [n_samples, n_features].
'''

'''
TF: Just counting the number of words in each document has 1 issue: it will give more weightage to
longer documents than shorter documents. To avoid this, we can use frequency (TF - Term Frequencies)
i.e. #count(word) / #Total words, in each document.
'''

'''
TF-IDF: Finally, we can even reduce the weightage of more common words like (the, is, an etc.) which
occurs in all document. This is called as TF-IDF i.e Term Frequency times inverse document frequency.
'''
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape) # (11314, 130107)

'''
There are various algorithms which can be used for text classification. We will start with the most
simplest one ‘Naive Bayes
You can easily build a NBclassifier in scikit using below 2 lines of code
'''

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

'''
All the code above can be replicated in less code by using a pipeline
'''
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()),
                     ])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

'''
Performance of naive bayes
'''
import numpy as np
twenty_test = fetch_20newsgroups(data_home='data', subset='test', shuffle=True)
predicted = text_clf.predict(twenty_test.data)
NBaccuracy = np.mean(predicted == twenty_test.target)
print("naive bayes accuracy", NBaccuracy)

'''
Creating a SVM classifier with pipeline
'''
from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge',
                                                   penalty='l2',
                                                   alpha=1e-3,
                                                   n_iter=5,
                                                   random_state=42)),
                        ])
_ = text_clf_svm.fit(twenty_train.data, twenty_train.target)
predicted_svm = text_clf_svm.predict(twenty_test.data)
svmAccuracy = np.mean(predicted_svm == twenty_test.target)
print("SVM Accuracy ", svmAccuracy)