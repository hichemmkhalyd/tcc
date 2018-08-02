import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import os
from time import time
import numpy as np
import pylab as pl
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import cross_validation
import itertools
import shutil
import matplotlib.pyplot as plt


from  sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessor import Preprocessor 



def benchmark():
    
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(X_train, y_train)
        clf_test()
        
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)
    
        
        


def clf_test():
        t0 = time()
        pred = clf.predict(X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)
    
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                                target_names=categories))
        
        
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))
        
        return metrics.f1_score(y_test, pred, average='micro')  
        

###############################################################################
#####################__Parâmetros pra rodar o algoritmo__######################
###############################################################################
clf = LinearSVC(loss='l2', penalty='l2',
                dual=False, tol=1e-3, class_weight='balanced')
#clf=svm.NuSVC(nu=0.7,kernel = 'linear',probability = True,class_weight = 'balanced',)
#clf=LinearSVC(loss='squared_hinge', penalty='l2',dual=False, tol=1e-3, class_weight='balanced')
DATA_FOLDER = "C:/Users/Seven/Documents/tcc_active learning/movie.csv"
ENCODING = 'utf-8'
result_x = []
result_y = []
###############################################################################






###############################################################################
################__Carrega dados dos arquivos + Preprocessamento__##############
###############################################################################        
 
df_data = pd.read_csv(DATA_FOLDER,encoding = ENCODING,header = 0)

df_neg = df_data[df_data.pol == 'Neg'].reset_index(drop=True)
df_pos = df_data[df_data.pol == 'Pos'].reset_index(drop=True)


df_train = pd.DataFrame(df_neg[0:1])
df_train = df_train.append(df_pos[0:1], ignore_index=True)
df_train = df_train.reindex(np.random.permutation(df_train.index))
df_train = df_train.reset_index(drop=True)

df_unlabeled = pd.DataFrame(df_neg[2:666])
df_unlabeled = df_unlabeled.append(df_pos[2:666], ignore_index=True)
df_unlabeled.rename(columns={'text':'bruto'}, inplace=True)
df_unlabeled = df_unlabeled.reindex(np.random.permutation(df_unlabeled.index))
df_unlabeled = df_unlabeled.reset_index(drop=True)

df_test = pd.DataFrame(df_neg[667:1000])
df_test = df_test.append(df_pos[667:1000], ignore_index=True)
df_test = df_test.reindex(np.random.permutation(df_test.index))
df_test = df_test.reset_index(drop=True)


    
categories = df_train.pol.unique()
       
pp = Preprocessor()
df_train.text   = pp.preprocess(df_train.text)
df_test.text    = pp.preprocess(df_test.text)
df_unlabeled    = df_unlabeled.assign(text = pp.preprocess(df_unlabeled.bruto))
###############################################################################

        



###############################################################################
##########################-->Loop do active learning<--########################
############################################################################### 
while True:
    
    print(df_train.pol.size)
    print(df_unlabeled.pol.size)
    print(df_test.pol.size)        
    y_train = df_train.pol
    y_test =  df_test.pol
    

    
    vectorizer = TfidfVectorizer(encoding= ENCODING, use_idf=True, norm='l2', binary=True, sublinear_tf=True,min_df=0.001, max_df=1.0, ngram_range=(1, 2), analyzer='word', stop_words=None)
    
    
    X_train     = vectorizer.fit_transform(df_train.text)
    X_test      = vectorizer.transform(df_test.text)
    X_unlabeled = vectorizer.transform(df_unlabeled.text)
    
    
    
    
    benchmark()    
    result_x.append(clf_test())
    result_y.append(df_train.pol.size)      
    
        
    if df_unlabeled.pol.size > 3:
        insert = {'pol':[],'text':[]}
        cont = 0
        for i in range(0, 4):
           insert["pol"].insert(cont,df_unlabeled.pol[i])
           insert["text"].insert(cont,df_unlabeled.text[i]) 
           cont+=1
           df_unlabeled = df_unlabeled.drop(i)
            
        
        df_insert = pd.DataFrame.from_dict(insert)
        df_train= df_train.append(df_insert, ignore_index=True)
        df_unlabeled = df_unlabeled.reset_index(drop=True)
        
        #labelNumber = input("Aperte qualquer tecla para próxima iteração")
        
    else:
        
        plt.plot(result_y, result_x, 'ro')
        plt.axis([0, 1400, 0.5, 0.90])
        plt.show()
        
        result = pd.DataFrame(result_y)
        result = result.assign(y=result_x)
        np.savetxt(r'C:/Users/Seven/Documents/tcc_active learning/resultados/movie_spv.txt', result, fmt='%f')
        
        break
