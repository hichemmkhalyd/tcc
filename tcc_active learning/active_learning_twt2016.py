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
    
        
      
        clf.fit(X_train, y_train)
        
        
        if ACTIVE == 2:
            y_unlabeled = clf.predict(X_unlabeled)
            y_unlabeled = pd.Series(y_unlabeled,name = "pol")
            y_unified   = y_train.append(y_unlabeled) 
            clf.fit(X_unified, y_unified)
       
        
        confidences = clf.decision_function(X_unlabeled)
        
        
        df = pd.DataFrame(clf.predict(X_unlabeled)) 
        df = df.assign(conf =  confidences.max(1))
        df.columns = ['pol','conf']
        
        df_pos = df[df.pol == 'positive']
        df_nlt = df[df.pol == 'neutral']
        df_neg = df[df.pol == 'negative']
        
        
        df_pos = df_pos.sort_values(by=['conf'],ascending = True)
        high_confidence_pos_samples = df_pos.conf.index[0:NUM_QUESTIONS]
        df_neg = df_neg.sort_values(by=['conf'],ascending = False)
        high_confidence_neg_samples = df_neg.conf.index[0:NUM_QUESTIONS]
        df_nlt = df_nlt.sort_values(by=['conf'],ascending = False)
        high_confidence_nlt_samples = df_nlt.conf.index[0:NUM_QUESTIONS]
        
        
        df_pos = df_pos.sort_values(by=['conf'],ascending = False)
        low_confidence_pos_samples = df_pos.conf.index[0:NUM_QUESTIONS]
        df_neg = df_neg.sort_values(by=['conf'],ascending = True)
        low_confidence_neg_samples = df_neg.conf.index[0:NUM_QUESTIONS]
        df_nlt = df_nlt.sort_values(by=['conf'],ascending = True)
        low_confidence_nlt_samples = df_nlt.conf.index[0:NUM_QUESTIONS]
        
        
        
        question_samples = []
        question_samples.extend(high_confidence_pos_samples.tolist())
        question_samples.extend(high_confidence_neg_samples.tolist())
        question_samples.extend(high_confidence_nlt_samples.tolist())
        
        question_samples.extend(low_confidence_pos_samples.tolist())
        question_samples.extend(low_confidence_neg_samples.tolist())
        question_samples.extend(low_confidence_nlt_samples.tolist())
        
        
        return question_samples


def clf_test():
       
        pred = clf.predict(X_test)
        
       
    
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
                dual=False, tol=1e-3)
#clf=svm.NuSVC(nu=0.7,kernel = 'linear',probability = True,class_weight = 'balanced',)
#clf=LinearSVC(loss='squared_hinge', penalty='l2',dual=False, tol=1e-3, class_weight='balanced')
NUM_QUESTIONS = 2
ACTIVE = 1
PATH_TRAIN = "C:/Users/Seven/Documents/tcc_active learning/dados/twt_2016/2016train.csv"
PATH_TEST  = "C:/Users/Seven/Documents/tcc_active learning/dados/twt_2016/2016test.csv"
ENCODING = 'utf-8'
result_x = []
result_y = []
###############################################################################






###############################################################################
################__Carrega dados dos arquivos + Preprocessamento__##############
###############################################################################        
 
df_train = pd.read_csv(PATH_TRAIN,encoding = ENCODING,header = 0)
df_test = pd.read_csv(PATH_TEST,encoding = ENCODING,header = 0)


df_neg = df_train[df_train.pol == 'negative'].reset_index(drop=True)
df_pos = df_train[df_train.pol == 'positive'].reset_index(drop=True)
df_nlt = df_train[df_train.pol == 'neutral'].reset_index(drop=True)



df_train = pd.DataFrame(df_neg[0:1])
df_train = df_train.append(df_pos[0:1], ignore_index=True)
df_train = df_train.append(df_nlt[0:1], ignore_index=True)
df_train = df_train.reindex(np.random.permutation(df_train.index))
df_train = df_train.reset_index(drop=True)


df_unlabeled = pd.DataFrame(df_neg[2:df_neg.size])
df_unlabeled = df_unlabeled.append(df_pos[2:df_pos.size], ignore_index=True)
df_unlabeled = df_unlabeled.append(df_nlt[2:df_nlt.size], ignore_index=True)
df_unlabeled.rename(columns={'text':'bruto'}, inplace=True)
df_unlabeled = df_unlabeled.reindex(np.random.permutation(df_unlabeled.index))
df_unlabeled = df_unlabeled.reset_index(drop=True)

    
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
    

    
    vectorizer = TfidfVectorizer(encoding= ENCODING, use_idf=True, norm='l2', binary=False, sublinear_tf=True,min_df=0.0001, max_df=1.0, ngram_range=(1, 3), analyzer='word', stop_words=None)
    
    
    X_train     = vectorizer.fit_transform(df_train.text)
    X_test      = vectorizer.transform(df_test.text)
    X_unlabeled = vectorizer.transform(df_unlabeled.text)
    
    
    df_unified = df_train.append(df_unlabeled)
    X_unified  = vectorizer.transform(df_unified.text)
        
    
    
    
    
    question_samples = benchmark()    
    result_x.append(clf_test())
    result_y.append(df_train.pol.size)      
    
        
    if df_train.pol.size <2000:
        insert = {'pol':[],'text':[]}
        cont = 0
        for i in question_samples:
            '''
            print ('**************************content***************************')
            print (df_unlabeled.bruto[i])
            print ('**************************content end***********************')
            print ("Annotate this text (select one label):")
            for j in range(0, len(categories)):
                print ("%d = %s" %(j+1, categories[j]))
            labelNumber = input("Enter the correct label number:")
            while labelNumber.isdigit()== False:
                for j in range(0, len(categories)):
                     print ("%d = %s" %(j+1, categories[j]))
                labelNumber = input("Enter the correct label number (a number please):")
            labelNumber = int(labelNumber)
            category = categories[labelNumber - 1] 
            insert["pol"].insert(cont,category)
            '''
            try:
                insert["pol"].insert(cont,df_unlabeled.pol[i])
                insert["text"].insert(cont,df_unlabeled.text[i]) 
                cont+=1
                df_unlabeled = df_unlabeled.drop(i)
            except:
                print ("This is an error message!")
            
        
        df_insert = pd.DataFrame.from_dict(insert)
        df_train= df_train.append(df_insert, ignore_index=True)
        df_unlabeled = df_unlabeled.reset_index(drop=True)
        
        #labelNumber = input("Aperte qualquer tecla para próxima iteração")
        
    else:
        result_y_active = result_y
        result_x_active = result_x
        plt.plot(result_y_active, result_x_active,label ='Active learning')
        #plt.plot(result_y_spv, result_x_spv,label = 'Convencional')
        plt.axis([0, 2000, 0.3, 0.6])
        plt.legend(loc='lower right', shadow=True, fontsize='x-large')
        plt.grid(True)
        plt.xlabel('Número de instâncias no conjunto de treino')
        plt.ylabel('f1-score')
        plt.title('twitter-2015')
        plt.show()
        
        result = pd.DataFrame(result_y)
        result = result.assign(y=result_x)
        np.savetxt(r'C:/Users/Seven/Documents/tcc_active learning/resultados/2013_active.txt', result, fmt='%f')
        
        break
