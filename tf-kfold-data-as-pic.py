#https://doi.org/10.1007/s10618-019-00619-1 time series and deep learning 

##########################
#s'ocupper de maxsize!!!!!
###########################


import numpy as np
import random
from numpy import array
import itertools
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc,precision_recall_curve
import matplotlib.pyplot as plt
from numpy import interp
from itertools import cycle
import json
import matplotlib.colors as mcolors
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import tensorflow as tf
from PIL import Image as img


from lbcnn2 import buildModel,returnName

#17/11/20 trying to keep timescale....
#1/12/20 with functional API

import pandas as pd
import os
import datetime

modelname=returnName()
featList=["tofcou","t4t1",'hr'] #NIBPM,'NIBPM','SP02'
features=len(featList)
classes=3
global_accuracy=0
global_loss=0
epochs= 100 #500
splits=10
picSize=768

modeltime=""
now = datetime.datetime.now()
dset=[]

dataset_y=[] #0 nothing, 1 robinul, 2 bridion
dataset_y2=[]
index=0
filecount=0
largest=0
filenames=[]
F1List=[]
filelist=os.listdir('./good-s/')
for inputfile in filelist:
 filenames.append(inputfile)
 if inputfile.endswith('.csv_2'):
  filecount+=1
  if ('NOTHING') in inputfile:
   dataset_y.append([1,0,0])
   dataset_y2.append(0)
  elif ('ROBIN') in inputfile:
   #continue
   dataset_y.append([0,1,0])
   dataset_y2.append(1)
  elif ('BRIDIO') in inputfile:
   #continue
   dataset_y.append([0,0,1])
   dataset_y2.append(2)
  df=pd.read_csv('./good-s/'+inputfile)
  #if 'NIBPM' not in df.columns.to_list():
  # try: 
  #  df['NIBPM']=df['APM']
  # except:
  #  print(inputfile)
  temp=df[featList].copy()
  temp['t4t1'] = temp.apply(
    lambda row: 100 if row['t4t1']>100 else row['t4t1'],
    axis=1)
  temp.fillna(0)
  temp["hr"]=(temp["hr"]-temp["hr"].min())/(temp["hr"].max()-temp["hr"].min())
  if len(temp)>largest:
   largest=len(temp)
  dset.append(np.array(temp).astype(np.float64))
  index+=1
dataset_y=tf.keras.utils.to_categorical(dataset_y2)
result_data=[]
z=0
imgDset=[]
for j in dset:
 image=[]
 for i in range(np.nan_to_num(j.shape[1])):
  min=np.nan_to_num(j)[:,i].min()
  max=np.nan_to_num(j)[:,i].max()
  maxmin=max-min
  if (maxmin==0):
   maxmin=1
  image.append((np.nan_to_num(j)[:,i]*255/(maxmin)).tolist())
 #imgDset.append(image)
 image2=img.fromarray(np.array(image)).convert('L').resize((picSize,features*6),resample=img.LANCZOS)
 image2.save(f"imgs/{z}.png")
 imgt=tf.keras.utils.img_to_array(image2)
 imgt=imgt.reshape(imgt.shape[1],imgt.shape[0])#####,imgt.shape[2])
 imgDset.append(imgt)#img_to_array(image2))
 z+=1
imgDset=np.array(imgDset)
#imgDset=imgDset.reshape(imgDset.shape[1],imgDset.shape[0],imgDset.shape[2])
#print(f'shape {imgDset.shape[0]} {imgDset.shape[1]} {imgDset.shape[2]}')


dataset=imgDset
dataset_y=np.array(dataset_y).reshape(index,classes) #was 3
result_test=[]
seed=7
cvscores=[]
histories=[]
probabilities=[]
true_table=[]
 
modeltime=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
dirname1="{}-{}".format(modelname,modeltime)
if not os.path.exists(dirname1):
  os.makedirs('{}'.format(dirname1))
fold=0
fold_no=1

accuracies=[]
losses=[]

y2=pd.DataFrame(dataset_y2)
classesWeights={0:1.0}
classesWeightsList=[]

for i in range(classes):
 
 classesWeightsList.append((y2==i).sum())
divider=len(dataset_y2)/classesWeightsList[0]
classesWeights[1]= float((len(dataset_y2)/classesWeightsList[1])/divider)
classesWeights[2]= float((len(dataset_y2)/classesWeightsList[2])/divider)

kfold = KFold(n_splits=splits, shuffle=True, random_state=seed) #beware, n_splits too low
for train, test in kfold.split(np.array(dataset), np.array(dataset_y2)):



 dirname="{}/{}".format(dirname1,fold)
 fold+=1
 if not os.path.exists(dirname):
   os.makedirs('{}'.format(dirname))

 filepath_los=modeltime+"-{epoch:02d}-loss-{val_loss:.4f}.hdf5"
 filepath_acc=modeltime+"-{epoch:02d}-acc-{val_categorical_accuracy:.3f}.hdf5"
#here are the callbacks

 check_loss = tf.keras.callbacks.ModelCheckpoint(
    filepath_los, monitor='val_loss', 
    verbose=0,        
    save_best_only=True,        
    mode='min'
 )    

 check_accuracy = tf.keras.callbacks.ModelCheckpoint(
    filepath_acc, monitor='val_categorical_accuracy', save_best_only=True, 
    mode='max'
 )

 es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=20, min_delta=0.01,restore_best_weights=True) #patience was 500 
#patience was 100
#end callbacks


 X=[]
 Y=[]
 X_test=[]
 Y_test=[]
 Y_true=[]
 for c in train:	
  X.append(dataset[c])
  Y.append(dataset_y[c])
 for c in test:
  X_test.append(dataset[c])
  Y_test.append(dataset_y[c])
  Y_true.append(dataset_y2[c])
  
 X=np.array(X)
 Y=np.array(Y)
 X_test=np.array(X_test)
 Y_test=np.array(Y_test)
 print ("{}{}".format(largest,len(X)))
 train_len=len(X)
 
 stat_file=open("{}/stats.txt".format(dirname),"a")
 trainFile=open("{}/train.txt".format(dirname),"a")
#  trainFile.write(filenames)
#  trainFile.write(test)
  
 for i in test:
   trainFile.write("{}\n".format(filenames[i]))
 trainFile.close()


 model=buildModel(picSize,features,classes)




 stat_file.write("{}".format(dirname))
 ####stat_file.write("{}".format(model.to_yaml()))
 stat_file.flush()

 model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.SGD(lr=0.005), metrics=['categorical_accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()]) 
 #inputs=[X[:,:,1].reshape(X.shape[0],largest,1),X[:,:,2].reshape(X.shape[0],largest,1),X[:,:,0].reshape(X.shape[0],largest,1)]
 #t_inputs=[X_test[:,:,1].reshape(X_test.shape[0],largest,1),X_test[:,:,2].reshape(X_test.shape[0],largest,1),X_test[:,:,0].reshape(X_test.shape[0],largest,1)]
 
 inputs=X
 t_inputs=X_test
 print ("X shape {}, y shape {}".format(inputs.shape,Y.shape))
 history=model.fit(inputs,Y, batch_size =train_len//100, epochs = epochs,class_weight=classesWeights, validation_data=(t_inputs, Y_test),callbacks=[es]) #,check_accuracy,check_loss])
 #
 scores = model.evaluate(t_inputs, Y_test, verbose=0)
 for j in range (len(scores)):
  print(f'Score for fold {fold_no}: {model.metrics_names[j]} of {scores[j]}')
 precision=-1
 recall=-1
 for j in range(len(model.metrics_names)):
  a=model.metrics_names[j]
  if 'precis' in a:
   precision=scores[j]
  if 'recal' in a:
   recall=scores[j]
 if(precision==-1 or recall==-1):
  print (f'Could not find precision {precision} or recall {recall}')
 else:
  f1=2*(precision*recall)/(precision+recall)
  F1List.append(f1)
  print (f'F1 {f1}')
  stat_file.write('\nF1 is {}\n'.format(f1))
 accuracies.append(scores[1])
 losses.append(scores[0])


 stat_file.write('\nlowest val_loss is {}\n'.format(scores[0]))
 stat_file.write('highest val_cat_acc is {}\n'.format(scores[1]))
 
 
 stat_file.flush()
 histories.append(history.history)

########################
#plot accuracy and loss#
########################
 lw = 2
 colors=['aqua', 'darkorange', 'cornflowerblue','navy']
 plt.figure()
 stoppedepoch=epochs
 if es.stopped_epoch!=0:
  stoppedepoch=es.stopped_epoch+1

 for i,key in zip(range(len(history.history.keys())),history.history.keys()):
   plt.plot(range(stoppedepoch), history.history[key], color=list(mcolors.TABLEAU_COLORS.keys())[i], lw=lw,
             label=key)

 plt.xlim([0, stoppedepoch])
# plt.ylim([0.0, 1.05])
 plt.xlabel('Epoch')
 plt.ylabel('Accuracy and loss')
 plt.title('Accuracy and loss')
 plt.legend(loc="lower right")
# plt.show()
 plt.savefig('{}/accloss.png'.format(dirname))
 plt.close();


##################sklearn metrics time



# for i in histories[z].history.keys():
#  print (i)
#  print (histories[z].history[i])


#dataframe=pd.DataFrame(history.history)
#filedate=now.strftime("%Y-%m-%d-%H-%M")
#dataframe.to_csv("metric-{}.csv".format(filedate) )

#train_acc = model.evaluate(dataset, dataset_y)
#test_acc = model.evaluate(testset, testset_y)
#print (test_acc)

#for i in history.history.keys():
# print (i+':'+str(history.history[i][-1]))

 
#loading the model.... 
 #loaded_model = load_model('model.h5')
 loaded_model=model
 model.save('{}/model.h5'.format(dirname))
 del (model)
 del (history)

 final_prob_y=loaded_model.predict(t_inputs) #was predict_proba
# for z in history.history.keys():
#  print (history.history[z])

 scores = loaded_model.evaluate(t_inputs, Y_test)
 cvscores.append(scores)
 probabilities.append(final_prob_y)
 true_table.append(Y_true)
 fpr = dict()
 tpr = dict()
 roc_auc = dict()
 
 precision = dict()
 recall = dict()
 thresholds = dict()
 precision_recall_auc = dict()
 prob_true=dict()
 prob_pred=dict()
 brier_score=dict()

 for i in range(classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], final_prob_y[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
#compute pr auc curves
    recall[i],precision[i], thresholds[i] = precision_recall_curve(Y_test[:, i],final_prob_y[:, i])
    precision_recall_auc[i] = auc(precision[i],recall[i])
#calibration curves   
    prob_true[i], prob_pred[i] = calibration_curve(Y_test[:, i],final_prob_y[:, i], n_bins=10)
#brier_score
    brier_score[i]=brier_score_loss(Y_test[:, i],final_prob_y[:, i])
    
# Compute micro-average ROC curve and ROC area
 fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), final_prob_y.ravel())
 roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#compute pr curves
 recall["micro"],precision["micro"], thresholds["micro"] = precision_recall_curve(Y_test.ravel(), final_prob_y.ravel())
 precision_recall_auc["micro"] = auc(precision["micro"], recall["micro"])

 
 
 plt.figure()
 plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)
 all_fpr = np.unique(np.concatenate([fpr[i] for i in range(classes)]))#was 3

# Then interpolate all ROC curves at this points
 mean_tpr = np.zeros_like(all_fpr)
 for i in range(classes): #was 3
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
 mean_tpr /= classes #was 3

 fpr["macro"] = all_fpr
 tpr["macro"] = mean_tpr
 roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

 plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
 colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
 for i, color in zip(range(classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
 plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
 plt.xlim([0.0, 1.0])
 plt.ylim([0.0, 1.05])
 plt.xlabel('False Positive Rate')
 plt.ylabel('True Positive Rate')
 plt.title('Receiver operating characteristic example')
 plt.legend(loc="lower right")
# plt.show()
 plt.savefig('{}/roc.png'.format(dirname))
 plt.close();
 
 ############################PR curves##############################################
 
 plt.figure()
 plt.plot(precision["micro"], recall["micro"],
         label='micro-average PR curve (area = {0:0.2f})'
               ''.format(precision_recall_auc["micro"]),
         color='deeppink', linestyle=':', lw=1)
 all_precision = np.unique(np.concatenate([precision[i] for i in range(classes)]))

# Then interpolate all ROC curves at this points
 mean_recall = np.zeros_like(all_precision)
 for i in range(classes): #was 3
    mean_recall += interp(all_precision, precision[i], recall[i])

# Finally average it and compute AUC
 mean_recall /= classes #was 3

 #precision["macro"] = all_precision
 #recall["macro"] = mean_recall
 #precision_recall_auc["macro"] = auc(precision["macro"],recall["macro"])

 #plt.plot(precision["macro"], recall["macro"],
  #       label='macro-average PR curve (area = {0:0.2f})'
  #             ''.format(precision_recall_auc["macro"]),
  #       color='navy', linestyle=':', linewidth=4)
 colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
 for i, color in zip(range(classes), colors):
    plt.plot( precision[i],recall[i], color=color, lw=lw,
             label='PR curve of class {0} (area = {1:0.2f})'
             ''.format(i, precision_recall_auc[i]))
 #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
 plt.xlim([0.0, 1.0])
 plt.ylim([0.0, 1.05])
 plt.xlabel('Recall')
 plt.ylabel('Precision')
 plt.title('PR example')
 plt.legend(loc="lower right")
# plt.show()
 plt.savefig('{}/pr.png'.format(dirname))
 plt.close();
################################Calibration curves##############################################
 plt.figure()

 for i, color in zip(range(classes), colors):
    plt.plot( prob_pred[i],prob_true[i], color=color, lw=1, linestyle=':',marker='o',
             label='Prob true cal of class {0} brier:{1:0.3f}'
             ''.format(i,brier_score[i]))
 plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle=':')
 plt.xlim([0.0, 1.0])
 plt.ylim([0.0, 1.05])
 #plt.xlabel('Precision')
 #plt.ylabel('Recall')
 plt.title('Cal example')
 plt.legend(loc="lower right")
# plt.show()
 plt.savefig('{}/cal.png'.format(dirname))
 plt.close();
 stat_file.write("brier score {} {} {}\n".format(brier_score[0],brier_score[1],brier_score[2]))

 
 
 

 stat_file.close()
 
loss="loss: {} +/- {}".format(np.mean(losses),np.std(losses))
accuracy="accuracy: {} +/- {}".format(np.mean(accuracies),np.std(accuracies))
f1score="F1: {} +/- {}".format(np.mean(F1List),np.std(F1List))
print (accuracy)
print (loss)
print(f1score)

with open("{}/stats.txt".format(dirname1),"w") as f:
 f.write(loss)
 f.write(accuracy)
 f.write(f1score)
 
  
