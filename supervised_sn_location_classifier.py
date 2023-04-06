# http://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/
# This is the general format I'm going to use for the BOW classifier for the detecting bias project.
import pandas as pd
import numpy
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn import feature_selection
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import scikitplot
import seaborn as sn
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import MultinomialNB
# Import modules for evaluation purposes
# Import libraries for predcton
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc, f1_score
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Load in Data


raw_annotated_data = pd.read_csv("sample_location_sn_posts_dw_first_750.csv")
#unique_annotated_data = raw_annotated_data.drop_duplicates(subset=['Sentence', 'user_name'])
#unique_annotated_data.to_csv("sample_location_sn_posts_revised_750.csv")
drew_annotated = pd.read_csv("sample_location_sn_posts_revised_750.csv")
first_100 = drew_annotated.head(100)
first_100.to_csv("100_sn_location_annotation_sudeshna.csv")
gold_standard = unique_annotated_data.dropna(subset=['Annotation_source'])


#Make it =1 , then Source,
# if anything else, =0 not source

source_mapping = {1: '1',
                  2:'0',
                  3:'0',
                  4:'1',}
gold_standard = gold_standard.assign(source = gold_standard.Annotation_source.map(source_mapping))

# if Annotation_source = 2, then User, if anything else, = 0, not user
user_mapping  = {1: '0',
                  2:'1',
                  3:'0',
                  4:'1',}
gold_standard = gold_standard.assign(user = gold_standard.Annotation_source.map(user_mapping))


# SPLIT
# 1. First, for HW 7, write a script which loads your labeled dataset and divides it into a training-
# test split. Consider the various methods we discussed in class, and think about how you want
# to approach – e.g. using k-fold within your training set as your validation step. If you decide
# not to include a validation step, note why in your write up. Push this script to github. Write a
# short paragraph in overleaf re: the choices you made and why you made them and submit your
# paragraph on canvas (can be included in the same doc as hw 8 below).


# X_train, X_test, y_train, y_test = train_test_split(gold_standard["Sentence"], gold_standard["quote_use"].values , test_size=0.20, random_state=0)
# Use k-fold
# Show the size of our datasets
# print('X Train Size:',X_train.shape)
# print('X Test Size:',X_test.shape)
skf = StratifiedKFold(n_splits=5)
X = gold_standard['Sentence']
y = gold_standard['source']

metrics = []

skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    vect = CountVectorizer(ngram_range=(1, 2), max_features=10000, stop_words="english")
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)

    metrics.append(accuracy_score(y_test, y_pred_class))
    metrics2 = []
# Create the confussion matrix
def plot_confussion_matrix(y_test, y_pred):
    ''' Plot the confussion matrix for the target labels and predictions '''
    cm = confusion_matrix(y_test, y_pred)

    # Create a dataframe with the confussion matrix values
    df_cm = pd.DataFrame(cm, range(cm.shape[0]),
                         range(cm.shape[1]))
    # plt.figure(figsize = (10,7))
    # Plot the confussion matrix
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, fmt='.0f', annot_kws={"size": 10})  # font size
    plt.show()


def plot_roc_curve(y_test, y_pred):
    ''' Plot the ROC curve for the target labels and predictions'''
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


print('Mean accuracy: ', numpy.mean(metrics, axis=0))
print('Std for accuracy: ', numpy.std(metrics, axis=0))
print(classification_report(y_test, y_pred_class, digits=5))
print(accuracy_score(y_test, y_pred_class))
scikitplot.metrics.plot_confusion_matrix(y_test,  y_pred_class)

# try multiple ways of calculating features
# Create the numericalizer TFIDF for lowercase
# tfidf = TfidfTransformer(encoding = "utf-8")
# Numericalize the train dataset
# tf_idf_train = tfidf.fit_transform(X_train.values.astype('U'))
# Numericalize the test dataset
# tf_idf_test = tfidf.transform(X_test.values.astype('U'))

# pipe = Pipeline([('count', CountVectorizer(vocabulary=vocab)),('tfid', TfidfTransformer())]).fit(X_train)


model = MultinomialNB()
model.fit(X_train_dtm, y_train)
y_pred_class2 = model.predict(X_test_dtm)
print(classification_report(y_test,y_pred_class2))
print(accuracy_score(y_test, y_pred_class2))


rf_model = RandomForestClassifier(random_state=1)
rf_model.fit(X_train_dtm, y_train)
y_pred_class_rf = rf_model.predict(X_test_dtm)
print(classification_report(y_test,y_pred_class_rf))
print(accuracy_score(y_test, y_pred_class_rf))
plot_confussion_matrix(y_test, y_pred_class_rf)
plot_roc_curve(y_test, y_pred_class_rf)
scikitplot.metrics.plot_confusion_matrix(y_test,  y_pred_class_rf)





y_pred = model.predict(X_test_dtm)

plot_confussion_matrix(y_test, y_pred)
plot_roc_curve(y_test, y_pred)

# BOW MODEL
# 2. Next, for HW 8, write a script that trains a BOW model – could be any non-neural model you
# like. Consider at least two variation of the features of your model—e.g. using counts v. TF-IDF
# representations of your text—rather than two model types. Write up a paragraph describing the
# choices you made in your overleaf doc. Consider the many available tutorials for this sort of thing
# (e.g. here and the sklearn package). Finally, estimate the F1 score and plot a precision/recall
# curve with only one model specification plotted. Consider using a tutorial (e.g. here). Include the
# plot in your write up on overleaf and submit on Canvas.


clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()

print('5-fold cross validation:\n')

labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes']

for clf, label in zip([clf1, clf2, clf3], labels):
    scores = model_selection.cross_val_score(clf, X_train_dtm, y_train,
                                             cv=5,
                                             scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))

# Ensemble voting

from mlxtend.classifier import EnsembleVoteClassifier

eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[1, 1, 1])

labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Ensemble']
for clf, label in zip([clf1, clf2, clf3, eclf], labels):
    scores = model_selection.cross_val_score(clf, X_train_dtm, y_train),
    cv = 5,
    scoring = 'accuracy'
    print("Accuracy: %0.2f (+/- %0.2f) [%s]", scores.mean(), scores.std(), label)