# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
Article:
https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568#:~:text=Linear%20Support%20Vector%20Machine%20is,the%20best%20text%20classification%20algorithms.&text=We%20achieve%20a%20higher%20accuracy,5%25%20improvement%20over%20Naive%20Bayes.

INterpretar resultados:
https://www.iartificial.net/precision-recall-f1-accuracy-en-clasificacion/

"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scikitplot.metrics import plot_roc
from scikitplot.metrics import plot_precision_recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import class_weight
#Classifier Models
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
#Bayes
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
#LSVM
from sklearn.linear_model import SGDClassifier
#Logistic Regression
from sklearn.linear_model import LogisticRegression
#OverSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss

import seaborn as sns



def arpic_describe_data(df):
    """ Describe el set de datos base a analizar en este proyecto. """
    print(df['TIPO'].value_counts()*10)
    plt.figure(figsize=(8,6))
    df.groupby('TIPO').COMENTARIO.count().plot.bar(ylim=0)
    #plt.show()

def arpic_plot_data(y_train,my_tags):
    count = y_train.value_counts()
    count.plot.bar()
    plt.ylabel('Number of records')
    plt.xlabel('Target Class')
    plt.show()
    
def arpic_plot_roc(y_test, y_score):
    plot_roc(y_test, y_score)
    plt.show()  
    
def arpic_plot_precision(y_test, y_score):
    plot_precision_recall(y_test, y_score)
    plt.show()     
    
def arpic_plot_heatmap(y_train, conmat, lb):
    val = np.mat(conmat) 
    classnames = list(set(y_train))
    df_cm = pd.DataFrame(
    
        val, index=classnames, columns=classnames, 
    
    )
    #print(df_cm)
    plt.figure()
    df_cm = df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis]  
    heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues", vmin=0, vmax=1)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(lb)
    plt.show()     


def arpic_DecisionTreeClassifier(X_train, X_test, y_train, y_test, my_tags, class_weight=None):  
    """ Clasificacion por DecisionTreeClassifier. """
    #xsarpic_plot_data(y_train, my_tags)
    
    # Calculate weight
    if class_weight:
        model = DecisionTreeClassifier(class_weight=class_weight)
    else:
        model = DecisionTreeClassifier()
        
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)
    y_pred = model.predict(X_test)    
    conmat = confusion_matrix(y_test, y_pred)
    arpic_plot_data(y_train, my_tags)
    arpic_plot_roc(y_test, y_score)
    arpic_plot_precision(y_test, y_score)
    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred,target_names=my_tags)) 
    return y_score, y_pred, conmat


def arpic_bayes(X_train, X_test, y_train, y_test, my_tags, balancing=False):  
    """ Bayes """
    
    if balancing:
        print('SI')
        nb = Pipeline([
                        ('clf', MultinomialNB()),
                      ])
    else:
        print('NO')
        nb = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultinomialNB()),
                      ])
    nb.fit(X_train, y_train)
    y_score = nb.predict_proba(X_test)
    y_pred = nb.predict(X_test)
    conmat = confusion_matrix(y_test, y_pred)
    arpic_plot_roc(y_test, y_score)
    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred,target_names=my_tags))
    return y_score, y_pred, conmat
      

def arpic_lsvm(X_train, X_test, y_train, y_test, my_tags, balancing=False, class_weight=None):  
    """ lsvm """
    arpic_plot_data(y_train, my_tags)
    if balancing:
        print('SI')
        # Calculate weight
        if class_weight:
            sgd = Pipeline([
                            ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None,class_weight=class_weight)),
                          ])
        else:    
            sgd = Pipeline([
                            ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                          ])   
    else:
        print('NO')
        sgd = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                       ])
    sgd.fit(X_train, y_train)
    y_score = sgd.decision_function(X_test)
    y_pred = sgd.predict(X_test)
    conmat = confusion_matrix(y_test, y_pred)
    arpic_plot_roc(y_test, y_score)
    conmat = confusion_matrix(y_test, y_pred)
    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred,target_names=my_tags))
    return y_score, y_pred, conmat


def arpic_lr(X_train, X_test, y_train, y_test, my_tags, balancing=False, class_weight=None):
    """ Logistic Regression """ 
    arpic_plot_data(y_train, my_tags)
    if balancing:
        print('SI')
        # Calculate weight
        if class_weight:
            logreg = Pipeline([
                            ('clf',  LogisticRegression(n_jobs=1, C=1e5,class_weight=class_weight)),
                          ])
        else:
            logreg = Pipeline([
                            ('clf',  LogisticRegression(n_jobs=1, C=1e5)),
                          ])
    else:
        print('NO')
        logreg = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf',  LogisticRegression(n_jobs=1, C=1e5)),
                       ])
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)   
    y_score = logreg.predict_proba(X_test)
    conmat = confusion_matrix(y_test, y_pred)
    arpic_plot_roc(y_test, y_score)
    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred,target_names=my_tags)) 
    return y_pred, y_score, conmat


def arpic_RandomOverSampler(X_train, X_test, y_train, y_test): 
    over_sampler = RandomOverSampler(random_state=42)
    X_res, y_res = over_sampler.fit_resample(X_train, y_train)
    print('SOBREMUESTRANDO RandomOverSampler.')
    print(f"Training target statistics: {Counter(y_res)}")
    print(f"Testing target statistics: {Counter(y_test)}")
    return X_res, y_res
    #roc_auc_ros,fpr_ros,tpr_ros, _ = build_and_test(X_res, X_test, y_res, y_test)   


if __name__ == '__main__':
    #df = pd.read_csv('tango_data_3L.csv', encoding='utf-8')#COMENTARIO
    df = pd.read_csv('tango_data_limpia_3L.csv',encoding='utf-8')#COMENTARIO_CLEAN

    
    #Prepare vector
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(df.COMENTARIO)
    
    X = matrix
    Xp = df['COMENTARIO']
    y = df['TIPO']
    my_tags = ['VENTA','PETICION','QUEJA']
    
    
    #Ver data
    arpic_describe_data(df)

    #Set datos 70/30 Entrenamiento/Test
    X_train, X_test, y_train, y_test = train_test_split(Xp, y, test_size=0.3, random_state=100)
    
    
    tfidf_wm = vectorizer.fit_transform(X_train)
    tfidf_tokens = vectorizer.get_feature_names()
    
    df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),index = ['VENTA','PETICION','QUEJA'],columns = tfidf_tokens)
    print(df_tfidfvect)
    
    """ DecisionTreeClassifier. """
    # #X
    # lb = "Balancing[NO] - Classifier[DecisionTreeClassifier]"
    # print(lb)
    # y_pred, y_score, conmat = arpic_DecisionTreeClassifier(X_train, X_test, y_train, y_test, my_tags)
    # arpic_plot_heatmap(y_test,conmat,lb)
    
    """ DecisionTreeClassifier - RandomOverSampler """ 
    # #X
    # lb = "Balancing[RandomOverSampler] - Classifier[DecisionTreeClassifier]"
    # print(lb)
    # over_sampler = RandomOverSampler(random_state=42)
    # X_res, y_res = over_sampler.fit_resample(X_train, y_train)
    # # arpic_plot_data(y_res, my_tags)
    # y_pred, y_score, conmat = arpic_DecisionTreeClassifier(X_res, X_test, y_res, y_test, my_tags)  
    # arpic_plot_heatmap(y_test,conmat,lb)
    
    """ DecisionTreeClassifier - smote """ 
    # #X
    # lb = "Balancing[smote] - Classifier[DecisionTreeClassifier]"
    # print(lb)    
    # over_sampler = SMOTE(k_neighbors=2)
    # X_res, y_res = over_sampler.fit_resample(X_train, y_train)
    # y_pred, y_score, conmat = arpic_DecisionTreeClassifier(X_res, X_test, y_res, y_test, my_tags)
    # arpic_plot_heatmap(y_test,conmat,lb)

    """ DecisionTreeClassifier - RandomUnderSampler """ 
    # #X
    # lb = "Balancing[RandomUnderSampler] - Classifier[DecisionTreeClassifier]"
    # print(lb)        
    # under_sampler = RandomUnderSampler(random_state=42)
    # X_res, y_res = under_sampler.fit_resample(X_train, y_train)
    # y_pred, y_score, conmat = arpic_DecisionTreeClassifier(X_res, X_test, y_res, y_test, my_tags)  
    # arpic_plot_heatmap(y_test,conmat,lb)

    """ DecisionTreeClassifier - NearMiss """ 
    # #X
    # lb = "Balancing[NearMiss] - Classifier[DecisionTreeClassifier]"
    # print(lb) 
    # under_sampler = NearMiss()
    # X_res, y_res = under_sampler.fit_resample(X_train, y_train)
    # y_pred, y_score, conmat = arpic_DecisionTreeClassifier(X_res, X_test, y_res, y_test, my_tags)
    # arpic_plot_heatmap(y_test,conmat,lb)
    
    """ DecisionTreeClassifier - class_weight """ 
    # #X
    # lb = "Balancing[class_weight] - Classifier[DecisionTreeClassifier]"
    # print(lb) 
    # cw = class_weight.compute_class_weight( 'balanced', classes = np.unique(y_train), y = y_train )
    # wights = dict(zip(np.unique(y_train),cw))
    # y_pred, y_score, conmat = arpic_DecisionTreeClassifier(X_train, X_test, y_train, y_test, my_tags,class_weight=wights)
    # arpic_plot_heatmap(y_test,conmat,lb)  
    

    """ Bayes """
    # #Xp
    # lb = "Balancing[NO] - Classifier[Bayes]"
    # print(lb)
    # arpic_plot_data(y_train, my_tags)
    # y_pred, y_score, conmat = arpic_bayes(X_train, X_test, y_train, y_test, my_tags, balancing=False)
    # arpic_plot_heatmap(y_test,conmat,lb)
    
    """ Bayes - RandomOverSampler """ 
    # #X
    # lb = "Balancing[RandomOverSampler] - Classifier[Bayes]"
    # print(lb)
    # over_sampler = RandomOverSampler(random_state=42)
    # X_res, y_res = over_sampler.fit_resample(X_train, y_train)
    # arpic_plot_data(y_res, my_tags)
    # y_pred, y_score, conmat =  arpic_bayes(X_res, X_test, y_res, y_test, my_tags, balancing=True) 
    # arpic_plot_heatmap(y_test,conmat,lb)
    
    """ Bayes - smote """ 
    # #X
    # lb = "Balancing[smote] - Classifier[Bayes]"
    # print(lb)
    # over_sampler = SMOTE(k_neighbors=2)
    # X_res, y_res = over_sampler.fit_resample(X_train, y_train)
    # arpic_plot_data(y_res, my_tags)
    # y_pred, y_score, conmat =  arpic_bayes(X_res, X_test, y_res, y_test, my_tags, balancing=True) 
    # arpic_plot_heatmap(y_test,conmat,lb)
    
    """ Bayes - RandomUnderSampler """ 
    # #X
    # lb = "Balancing[RandomUnderSampler] - Classifier[Bayes]"
    # print(lb)
    # under_sampler = RandomUnderSampler(random_state=42)
    # X_res, y_res = under_sampler.fit_resample(X_train, y_train)
    # arpic_plot_data(y_res, my_tags)
    # y_pred, y_score, conmat =  arpic_bayes(X_res, X_test, y_res, y_test, my_tags, balancing=True)  
    # arpic_plot_heatmap(y_test,conmat,lb)
    
    """ Bayes - NearMiss """ 
    # #X
    # lb = "Balancing[NearMiss] - Classifier[Bayes]"
    # print(lb)
    # under_sampler = NearMiss()
    # X_res, y_res = under_sampler.fit_resample(X_train, y_train)
    # arpic_plot_data(y_res, my_tags)
    # y_pred, y_score, conmat =  arpic_bayes(X_res, X_test, y_res, y_test, my_tags, balancing=True)
    # arpic_plot_heatmap(y_test,conmat,lb)
        
    
    """ Linear Support Vector Machine """ 
    # #Xp
    # lb = "Balancing[NO] - Classifier[Lsvm]"
    # print(lb)  
    # y_pred, y_score, conmat = arpic_lsvm(X_train, X_test, y_train, y_test, my_tags)
    # arpic_plot_heatmap(y_test,conmat,lb)
    
    """ Linear Support Vector Machine - RandomOverSampler """ 
    # #X
    # lb = "Balancing[RandomOverSampler] - Classifier[Lsvm]"
    # print(lb)     
    # over_sampler = RandomOverSampler(random_state=42)
    # X_res, y_res = over_sampler.fit_resample(X_train, y_train)
    # y_pred, y_score, conmat = arpic_lsvm(X_res, X_test, y_res, y_test, my_tags, balancing=True) 
    # arpic_plot_heatmap(y_test,conmat,lb)     

    """ Linear Support Vector Machine - smote """ 
    # #X
    # lb = "Balancing[smote] - Classifier[Lsvm]"
    # print(lb)     
    # over_sampler = SMOTE(k_neighbors=2)
    # X_res, y_res = over_sampler.fit_resample(X_train, y_train)
    # y_pred, y_score, conmat =  arpic_lsvm(X_res, X_test, y_res, y_test, my_tags, balancing=True) 
    # arpic_plot_heatmap(y_test,conmat,lb)  
    
    """ Linear Support Vector Machine - RandomUnderSampler """ 
    # #X
    # lb = "Balancing[RandomUnderSampler] - Classifier[Lsvm]"
    # print(lb)
    # under_sampler = RandomUnderSampler(random_state=42)
    # X_res, y_res = under_sampler.fit_resample(X_train, y_train)
    # y_pred, y_score, conmat =  arpic_lsvm(X_res, X_test, y_res, y_test, my_tags, balancing=True) 
    # arpic_plot_heatmap(y_test,conmat,lb) 

    """ Linear Support Vector Machine - NearMiss """ 
    # #X
    # lb = "Balancing[NearMiss] - Classifier[Lsvm]"
    # print(lb)
    # under_sampler = NearMiss()
    # X_res, y_res = under_sampler.fit_resample(X_train, y_train)
    # y_pred, y_score, conmat =  arpic_lsvm(X_res, X_test, y_res, y_test, my_tags, balancing=True)
    # arpic_plot_heatmap(y_test,conmat,lb) 

    
    """ Linear Support Vector Machine - class_weight """   
    # #X
    # lb = "Balancing[class_weight] - Classifier[Lsvm]"
    # print(lb)  
    # cw = class_weight.compute_class_weight( 'balanced', classes = np.unique(y_train), y = y_train )
    # wights = dict(zip(np.unique(y_train),cw))
    # y_pred, y_score, conmat = arpic_lsvm(X_train, X_test, y_train, y_test, my_tags, balancing=True, class_weight=wights)
    # arpic_plot_heatmap(y_test,conmat,lb)     


    """ Logistic Regression """ 
    # #Xp
    # lb = "Balancing[NO] - Classifier[Logistic Regression]"
    # print(lb)    
    # y_pred, y_score, conmat = arpic_lr(X_train, X_test, y_train, y_test, my_tags) 
    # arpic_plot_heatmap(y_test,conmat,lb)
   
    """ Logistic Regression - RandomOverSampler """ 
    # #X
    # lb = "Balancing[RandomOverSampler] - Classifier[Logistic Regression]"
    # print(lb) 
    # over_sampler = RandomOverSampler(random_state=42)
    # X_res, y_res = over_sampler.fit_resample(X_train, y_train)
    # y_pred, y_score, conmat = arpic_lr(X_res, X_test, y_res, y_test, my_tags, balancing=True) 
    # arpic_plot_heatmap(y_test,conmat,lb) 
  
    """ Logistic Regression - smote """ 
    # #X
    # lb = "Balancing[smote] - Classifier[Logistic Regression]"
    # print(lb) 
    # over_sampler = SMOTE(k_neighbors=2)
    # X_res, y_res = over_sampler.fit_resample(X_train, y_train)
    # y_pred, y_score, conmat = arpic_lr(X_res, X_test, y_res, y_test, my_tags, balancing=True) 
    # arpic_plot_heatmap(y_test,conmat,lb)  

    """ Logistic Regression - RandomUnderSampler """ 
    # #X
    # lb = "Balancing[RandomUnderSampler] - Classifier[Logistic Regression]"
    # print(lb) 
    # under_sampler = RandomUnderSampler(random_state=42)
    # X_res, y_res = under_sampler.fit_resample(X_train, y_train)
    # y_pred, y_score, conmat = arpic_lr(X_res, X_test, y_res, y_test, my_tags, balancing=True) 
    # arpic_plot_heatmap(y_test,conmat,lb)  
        
    """ Logistic Regression - NearMiss """ 
    # #X
    # lb = "Balancing[NearMiss] - Classifier[Logistic Regression]"
    # print(lb) 
    # under_sampler = NearMiss()
    # X_res, y_res = under_sampler.fit_resample(X_train, y_train)
    # y_pred, y_score, conmat = arpic_lr(X_res, X_test, y_res, y_test, my_tags, balancing=True) 
    # arpic_plot_heatmap(y_test,conmat,lb)  
    
    """ Logistic Regression  - class_weight """ 
    # #X
    # lb = "Balancing[class_weight] - Classifier[Logistic Regression]"
    # print(lb)     
    # cw = class_weight.compute_class_weight( 'balanced', classes = np.unique(y_train), y = y_train )
    # wights = dict(zip(np.unique(y_train),cw))
    # y_pred, y_score, conmat = arpic_lr(X_train, X_test, y_train, y_test, my_tags, balancing=True, class_weight=wights)
    # arpic_plot_heatmap(y_test,conmat,lb) 
 
    
    
    
    
        