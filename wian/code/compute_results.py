
import numpy as np
import sys
import os
from os.path import dirname
import pandas as pd
import math
import datetime
import re
import cPickle as pickle
from collections import defaultdict

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
from funcsigs import signature

def results_svm(model, ids, X, Y, label, task, labels, out_f,args):
    criteria, _, clf = model

    # for AUC
    P_ = clf.decision_function(X)

    # sklearn has stupid changes in API when doing binary classification. make it conform to 3+
    if len(labels)==2:
        m = X.shape[0]
        P = np.zeros((m,2))
        P[:,0] = -P_
        P[:,1] =  P_
    else:
        P = P_

    # hard predictions
    train_pred = P.argmax(axis=1)

    # what is the predicted vocab without the dummy label?
    if task in ['los','age','sapsii']:
        V = range(len(labels))
    else:
        V = labels.keys()

    out_f.write('%s %s' % (unicode(label),task))
    out_f.write(unicode('\n'))
    if len(V) == 2:
        scores = P[:,1] - P[:,0]
        compute_stats_binary(task, train_pred, scores, Y, criteria, out_f, label , args)
    else:
        compute_stats_multiclass(task,train_pred,P,Y,criteria,out_f)
    out_f.write(unicode('\n\n'))



def results_onehot_keras(model, ids, X, onehot_Y, label, task, out_f, args):
    criteria, num_docs, lstm_model = model

    # for AUC
    P = lstm_model.predict(X)

    # hard predictions
    train_pred = P.argmax(axis=1)

    Y = onehot_Y.argmax(axis=1)
    num_tags = P.shape[1]

    out_f.write('%s %s' % (unicode(label),task))
    out_f.write(unicode('\n'))
    if num_tags == 2:
        scores = P[:,1] #- P[:,0]
        compute_stats_binary(task, train_pred, scores, Y, criteria, out_f, label, args)
    else:
        compute_stats_multiclass(task, train_pred, P, Y, criteria, out_f)
    out_f.write(unicode('\n\n'))


def results_onehot_keras_v2(model, ids, X, onehot_Y, label, task, out_f, args):
    criteria, num_docs, lstm_model = model

    # for AUC
    P = lstm_model.predict(X)

    # hard predictions
    train_pred = P.argmax(axis=1)

    Y = onehot_Y.argmax(axis=1)
    num_tags = P.shape[1]

    out_f.write('%s %s' % (unicode(label),task))
    out_f.write(unicode('\n'))
    if num_tags == 2:
        scores = P[:,1] #- P[:,0]
        compute_stats_binary_v2(task, train_pred, scores, Y, criteria, out_f, label, args, ids)
    else:
        compute_stats_multiclass(task, train_pred, P, Y, criteria, out_f)
    out_f.write(unicode('\n\n'))



def pr_curve_plot(y, y_score, args):
    precision, recall, _ = precision_recall_curve(y, y_score)
    area = auc(recall, precision)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})

    plt.figure(2)
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AUC={0:0.2f}'.format(
        area))

    string = 'auprc_clinicalbert_' + args.readmission_mode + '.png'

    plt.savefig(os.path.join(args.output_dir, string))


def vote_pr_curve(y, y_pred):


    precision, recall, thres = precision_recall_curve(y, y_pred)
    pr_thres = pd.DataFrame(data=list(zip(precision, recall, thres)), columns=['prec', 'recall', 'thres'])

    temp = pr_thres[pr_thres.prec > 0.799999].reset_index()

    rp80 = 0
    if temp.size == 0:
        print('Test Sample too small or RP80=0')
    else:
        rp80 = temp.iloc[0].recall
        print('Recall at Precision of 80 is {}', rp80)

    return rp80

def compute_stats_binary(task, pred, P, ref, labels, out_f, label, args):
    # santiy check
    # assert all(map(int,P>0) == pred)

    V = [0,1]
    n = len(V)
    assert n==2, 'sorry, must be exactly two labels (how else would we do AUC?)'
    conf = np.zeros((n,n), dtype='int32')
    for p,r in zip(pred,ref):
        conf[p][r] += 1

    out_f.write(unicode(conf))
    out_f.write(unicode('\n'))

    tp = conf[1,1]
    tn = conf[0,0]
    fp = conf[1,0]
    fn = conf[0,1]

    precision   = tp / (tp + fp + 1e-9)
    recall      = tp / (tp + fn + 1e-9)
    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)

    f1 = (2*precision*recall) / (precision+recall+1e-9)

    tpr =  true_positive_rate(pred, ref)
    fpr = false_positive_rate(pred, ref)

    accuracy = (tp+tn) / (tp+tn+fp+fn + 1e-9)

    out_f.write(unicode('\tspecificity %.3f\n' % specificity))
    out_f.write(unicode('\tsensitivty: %.3f\n' % sensitivity))

    auc_value = roc_auc_score(ref, P)
    out_f.write(unicode('\t\t\tauc: %.3f\n' % auc_value))

    out_f.write(unicode('\taccuracy:   %.3f\n' % accuracy   ))
    out_f.write(unicode('\tprecision:  %.3f\n' % precision  ))
    out_f.write(unicode('\trecall:     %.3f\n' % recall     ))
    out_f.write(unicode('\tf1:         %.3f\n' % f1         ))
    out_f.write(unicode('\tTPR:        %.3f\n' % tpr        ))
    out_f.write(unicode('\tFPR:        %.3f\n' % fpr        ))
    if label =='TEST':
        fpr, tpr, thresholds = roc_curve(ref, P)
        auc_score = auc(fpr, tpr)

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Val (area = {:.3f})'.format(auc_score))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')

        string = 'auroc_clinicalbert_'+args.readmission_mode+'.png'
        plt.savefig(os.path.join(args.output_dir, string))
        plt.show()

        pr_curve_plot(ref, P, args)

        rp80 = vote_pr_curve(ref, P)
        result = {
                # 'eval_loss': eval_loss,
                  'eval_accuracy': accuracy,
                  # 'global_step': global_step_check,
                  # 'training loss': train_loss / number_training_steps,
                  'RP80': rp80}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            # logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                # logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))



def compute_stats_binary_v2(task, pred, P, ref, labels, out_f, label, args, ids):
    if label=='TEST':
        df = pd.DataFrame({'pred_score':P,'ID':ids,'Label':ref})
        # df['pred_score'] = score
        df_sort = df.sort_values(by=['ID'])
        #score
        temp = (df_sort.groupby(['ID'])['pred_score'].agg(max)+df_sort.groupby(['ID'])['pred_score'].agg(sum)/2)/(1+df_sort.groupby(['ID'])['pred_score'].agg(len)/2)
        x = df_sort.groupby(['ID'])['Label'].agg(np.min).values
        # df_out = pd.DataFrame({'logits': temp.values, 'ID': x})


        ref = x
        P = temp.values

        fpr, tpr, thresholds = roc_curve(ref, P)
        auc_score = auc(fpr, tpr)

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Val (area = {:.3f})'.format(auc_score))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')

        string = 'auroc_clinicalbert_' + args.readmission_mode + '.png'
        plt.savefig(os.path.join(args.output_dir, string))
        plt.show()

        pr_curve_plot(ref, P, args)

        rp80 = vote_pr_curve(ref, P)
        result = {
            # 'eval_loss': eval_loss,
            # 'eval_accuracy': accuracy,
            # 'global_step': global_step_check,
            # 'training loss': train_loss / number_training_steps,
            'RP80': rp80}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            # logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                # logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


def compute_stats_multiclass(task, pred, P, ref, labels_map, out_f):
    # santiy check
    assert all(map(int,P.argmax(axis=1)) == pred)

    # confusion matrix
    V = set(range(P.shape[1]))
    n = len(set((V)))
    conf = np.zeros((n,n), dtype='int32')
    for p,r in zip(pred,ref):
        conf[p][r] += 1

    # task labels (for printing results)
    if task in ['sapsii', 'age', 'los']:
        labels = []
        labels_ = [0] + labels_map
        for i in range(len(labels_)-1):
            label = '[%s,%s)' % (labels_[i],labels_[i+1])
            labels.append(label)
    else:
        labels = [label for label,i in sorted(labels_map.items(), key=lambda t:t[1])]

    out_f.write(unicode(conf))
    out_f.write(unicode('\n'))

    # compute P, R, F1
    precisions = []
    recalls = []
    f1s = []
    out_f.write(unicode('\t prec  rec    f1   label\n'))
    for i in range(n):
        label = labels[i]

        tp = conf[i,i]
        pred_pos = conf[i,:].sum()
        ref_pos  = conf[:,i].sum()

        precision   = tp / (pred_pos + 1e-9)
        recall      = tp / (ref_pos + 1e-9)
        f1 = (2*precision*recall) / (precision+recall+1e-9)

        out_f.write(unicode('\t%.3f %.3f %.3f %s\n' % (precision,recall,f1,label)))

        # Save info
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall    = sum(recalls   ) / len(recalls   )
    avg_f1        = sum(f1s       ) / len(f1s       )
    out_f.write(unicode('\t--------------------------\n'))
    out_f.write(unicode('\t%.3f %.3f %.3f avg\n' % (avg_precision,avg_recall,avg_f1)))



def true_positive_rate(pred, ref):
    tp,fn = 0,0
    for p,r in zip(pred,ref):
        if p==1 and r==1:
            tp += 1
        elif p==0 and r==1:
            fn += 1
    return tp / (tp + fn + 1e-9)


def false_positive_rate(pred, ref):
    fp,tn = 0,0
    for p,r in zip(pred,ref):
        if p==1 and r==0:
            fp += 1
        elif p==0 and r==0:
            tn += 1
    return fp / (fp + tn + 1e-9)

