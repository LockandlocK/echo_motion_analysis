#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:41:30 2019

@author: degerli
"""
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix,roc_curve, auc
from models import * 
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
     
ap = argparse.ArgumentParser()
ap.add_argument('-gpu', '--gpu', default='0')
ap.add_argument('-view', '--view', default='Muti_views')
ap.add_argument('-data', '--data', default='Muti_views')
#ap.add_argument('-data', '--data', default='DataSplits')

args = vars(ap.parse_args())

os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']
if not os.path.exists(os.path.join(os.getcwd(),'output', 'matrices')): os.makedirs(os.path.join(os.getcwd(),'output', 'matrices'))

#MODEL = ['CNN']
MODEL = ['SVM', 'DT', 'KNN', 'RF', 'MLP']
#MODEL = ['SVM', 'DT', 'KNN', 'RF']
#MODEL = ['MLP']

#REFIT= ['AUC', 'Accuracy', 'Recall', 'F1-Score', 'Precision']

REFIT= ['Accuracy']

# Define the number of splits
num_splits = 5

# Create KFold object
kf = KFold(n_splits=num_splits)
np.random.seed(seed=3)
x_2CH = np.squeeze(np.load(os.path.join(os.path.join(os.getcwd(),args['data']), 'x_train_' + '2CH' + '.npy')))
x_4CH = np.squeeze(np.load(os.path.join(os.path.join(os.getcwd(),args['data']), 'x_test_' + '4CH' + '.npy')))
y_2CH = np.squeeze(np.load(os.path.join(os.path.join(os.getcwd(),args['data']), 'y_train_' + '2CH' + '.npy')))
y_4CH = np.squeeze(np.load(os.path.join(os.path.join(os.getcwd(),args['data']), 'y_test_' + '4CH' + '.npy')))

print(x_2CH.shape)
print(x_4CH.shape)
print(y_2CH.shape)
print(y_4CH.shape)


X = np.concatenate((x_2CH, x_4CH), axis=1)
print(X.shape)

Y = y_2CH

num_samples = X.shape[0]

# Generate shuffled indices
shuffled_indices = np.random.permutation(num_samples)

# Shuffle the arrays X and Y using the shuffled indices
shuffled_X = X[shuffled_indices]
shuffled_Y = Y[shuffled_indices]
X = shuffled_X
Y = shuffled_Y
# Create a KFold object
kf = KFold(n_splits=5)

# Initialize an empty list to hold our fold data
fold_X_train = []
fold_Y_train = []
fold_X_test = []
fold_Y_test = []
# Split the data into folds

for train_index, test_index in kf.split(X):
    print(len(train_index))
    print(len(test_index))
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    fold_X_train.append(X_train)
    fold_Y_train.append(Y_train)
    fold_X_test.append(X_test)
    fold_Y_test.append(Y_test)
# Convert the list of fold data into a numpy array
x_train = np.array(fold_X_train)
y_train = np.array(fold_Y_train)
x_test = np.array(fold_X_test)
y_test = np.array(fold_Y_test)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)


for f in range(0,5):
#for f, (train_index, test_index) in enumerate(kf.split(x_train)):
#for train_index, test_index in kf.split(x_train):

    print('fold: ' + str(f))
    for i in range(len(MODEL)):
    
        for j in range(len(REFIT)):            
            #Shuffle train data
            np.random.seed(seed=3)
            idx = np.random.permutation(len(x_train[f]))
            x_train[f], y_train[f] = x_train[f][idx], y_train[f][idx]
        
            if MODEL[i] == 'SVM':    
                best_parameters, best_model = SVM_train(x_train[f], y_train[f], REFIT[j])
                
                score = best_model.predict(x_test[f])
                probas_ = best_model.predict_proba(x_test[f])
           # Compute ROC curve and AUC
                fpr, tpr, thresholds = roc_curve(y_test[f], probas_[:, 1])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)

                CM = confusion_matrix(y_test[f], score)
                metrics = performance_metrics(CM)
                
                #Save the results
                text_file = open(os.path.join(os.path.join(os.getcwd(), 'output'), MODEL[i] + '_' + args['view'] + '.txt'), "a")
                text_file.write ("\n\n\n----------FOLD " + str(f) + "-------------\n")
                text_file.write("\n\n\nConfusion Matrix :" + str(CM) + "\n")
                text_file.write ("\nScoring:" + REFIT[j])
                text_file.write ("\nSensitivity:" + str(metrics[0]))
                text_file.write ("\nSpecificity:" + str(metrics[1]))
                text_file.write("\nPrecision:" + str(metrics[2]))
                text_file.write ("\nF1-Score:" + str(metrics[3]))
                text_file.write ("\nF2-Score:" + str(metrics[4]))   
                text_file.write("\nAccuracy:" + str(metrics[5]))
                text_file.write('\nBest paramters:' + str(best_parameters))
                text_file.close()
                    
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] +'_score_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), score)
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] + '_y_test_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), y_test[f])
            elif MODEL[i] == 'MLP':    
                best_parameters, best_model = MLP_train(x_train[f], y_train[f], REFIT[j])
                score = best_model.predict(x_test[f])
                
                probas_ = best_model.predict_proba(x_test[f])
           # Compute ROC curve and AUC
                fpr, tpr, thresholds = roc_curve(y_test[f], probas_[:, 1])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)

                CM = confusion_matrix(y_test[f], score)
                metrics = performance_metrics(CM)
                
                #Save the results
                text_file = open(os.path.join(os.path.join(os.getcwd(), 'output'), MODEL[i] + '_' + args['view'] + '.txt'), "a")
                text_file.write ("\n\n\n----------FOLD " + str(f) + "-------------\n")
                text_file.write("\n\n\nConfusion Matrix :" + str(CM) + "\n")
                text_file.write ("\nScoring:" + REFIT[j])
                text_file.write ("\nSensitivity:" + str(metrics[0]))
                text_file.write ("\nSpecificity:" + str(metrics[1]))
                text_file.write("\nPrecision:" + str(metrics[2]))
                text_file.write ("\nF1-Score:" + str(metrics[3]))
                text_file.write ("\nF2-Score:" + str(metrics[4]))   
                text_file.write("\nAccuracy:" + str(metrics[5]))
                text_file.write('\nBest paramters:' + str(best_parameters))
                text_file.close()
                    
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] +'_score_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), score)
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] + '_y_test_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), y_test[f])
                                        
            elif MODEL[i] == 'KNN':
                best_parameters, best_model = KNN_train(x_train[f], y_train[f], REFIT[j])
                score = best_model.predict(x_test[f])

                probas_ = best_model.predict_proba(x_test[f])
           # Compute ROC curve and AUC
                fpr, tpr, thresholds = roc_curve(y_test[f], probas_[:, 1])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)

                probas_ = best_model.predict_proba(x_test[f])
                CM = confusion_matrix(y_test[f], score)
                metrics = performance_metrics(CM)
                
                #Save the results
                text_file = open(os.path.join(os.path.join(os.getcwd(), 'output'), MODEL[i] + '_' + args['view'] + '.txt'), "a")
                text_file.write ("\n\n\n----------FOLD " + str(f) + "-------------\n")
                text_file.write("\n\n\nConfusion Matrix :" + str(CM) + "\n")
                text_file.write ("\nScoring:" + REFIT[j])
                text_file.write ("\nSensitivity:" + str(metrics[0]))
                text_file.write ("\nSpecificity:" + str(metrics[1]))
                text_file.write("\nPrecision:" + str(metrics[2]))
                text_file.write ("\nF1-Score:" + str(metrics[3]))
                text_file.write ("\nF2-Score:" + str(metrics[4]))   
                text_file.write("\nAccuracy:" + str(metrics[5]))
                text_file.write('\nBest paramters:' + str(best_parameters))
                text_file.close()
                
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] +'_score_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), score)
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] + '_y_test_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), y_test[f])
                                       
            elif MODEL[i] == 'DT':
                best_parameters, best_model = DT_train(x_train[f], y_train[f], REFIT[j])
                score = best_model.predict(x_test[f])


                probas_ = best_model.predict_proba(x_test[f])
           # Compute ROC curve and AUC
                fpr, tpr, thresholds = roc_curve(y_test[f], probas_[:, 1])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)

                #probas_ = best_model.predict_proba(x_test[f])
                CM = confusion_matrix(y_test[f], score)
                metrics = performance_metrics(CM)
                
                #Save the results
                text_file = open(os.path.join(os.path.join(os.getcwd(), 'output'), MODEL[i] + '_' + args['view'] + '.txt'), "a")
                text_file.write ("\n\n\n----------FOLD " + str(f) + "-------------\n")
                text_file.write("\n\n\nConfusion Matrix :" + str(CM) + "\n")
                text_file.write ("\nScoring:" + REFIT[j])
                text_file.write ("\nSensitivity:" + str(metrics[0]))
                text_file.write ("\nSpecificity:" + str(metrics[1]))
                text_file.write("\nPrecision:" + str(metrics[2]))
                text_file.write ("\nF1-Score:" + str(metrics[3]))
                text_file.write ("\nF2-Score:" + str(metrics[4]))   
                text_file.write("\nAccuracy:" + str(metrics[5]))
                text_file.write('\nBest paramters:' + str(best_parameters))
                text_file.close()
                
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] +'_score_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), score)
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] + '_y_test_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), y_test[f])
                                
            elif MODEL[i] == 'RF':
                best_parameters, best_model = RF_train(x_train[f], y_train[f], REFIT[j])
                score = best_model.predict(x_test[f])
                
             


                probas_ = best_model.predict_proba(x_test[f])
           # Compute ROC curve and AUC
                fpr, tpr, thresholds = roc_curve(y_test[f], probas_[:, 1])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)

                CM = confusion_matrix(y_test[f], score)
                metrics = performance_metrics(CM)
                
                #Save the results
                text_file = open(os.path.join(os.path.join(os.getcwd(), 'output'), MODEL[i] + '_' + args['view'] + '.txt'), "a")
                text_file.write ("\n\n\n----------FOLD " + str(f) + "-------------\n")
                text_file.write("\n\n\nConfusion Matrix :" + str(CM) + "\n")
                text_file.write ("\nScoring:" + REFIT[j])
                text_file.write ("\nSensitivity:" + str(metrics[0]))
                text_file.write ("\nSpecificity:" + str(metrics[1]))
                text_file.write("\nPrecision:" + str(metrics[2]))
                text_file.write ("\nF1-Score:" + str(metrics[3]))
                text_file.write ("\nF2-Score:" + str(metrics[4]))   
                text_file.write("\nAccuracy:" + str(metrics[5]))
                text_file.write('\nBest paramters:' + str(best_parameters))
                text_file.close()
                
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] +'_score_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), score)
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] + '_y_test_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), y_test[f])
                                
            elif MODEL[i] == 'CNN':      
                x_train = np.expand_dims(x_train, axis = -1)
                x_test = np.expand_dims(x_test, axis = -1)
                best_model, best_parameters = CNN_train(x_train[f], y_train[f], REFIT[j])    
                score = best_model.predict(x_test[f])
                
                probas_ = best_model.predict_proba(x_test[f])
                CM = confusion_matrix(y_test[f], score)
                metrics = performance_metrics(CM)
                
                #Save the results
                text_file = open(os.path.join(os.path.join(os.getcwd(), 'output'), MODEL[i] + '_' + args['view'] + '.txt'), "a")
                text_file.write ("\n\n\n----------FOLD " + str(f) + "-------------\n")
                text_file.write("\n\n\nConfusion Matrix :" + str(CM) + "\n")
                text_file.write ("\nScoring:" + REFIT[j])
                text_file.write ("\nSensitivity:" + str(metrics[0]))
                text_file.write ("\nSpecificity:" + str(metrics[1]))
                text_file.write("\nPrecision:" + str(metrics[2]))
                text_file.write ("\nF1-Score:" + str(metrics[3]))
                text_file.write ("\nF2-Score:" + str(metrics[4]))   
                text_file.write("\nAccuracy:" + str(metrics[5]))
                text_file.write('\nBest paramters:' + str(best_parameters))
                text_file.close()
                
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] +'_score_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), score)
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] + '_y_test_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), y_test[f])

mean_tpr = np.mean(tprs, axis=0)
#print(mean_fpr)
mean_tpr[-1] = 1.0
#mean_auc = auc(mean_fpr, mean_tpr)

#print(mean_fpr)
print(mean_tpr)

