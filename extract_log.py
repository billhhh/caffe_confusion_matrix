#
#Author: Bill Wang
#

import numpy as np
import re
import operator
import os
NUM_CAT = 724

ACCURACY_EXTRACT_FLAG = 1
CONFUSION_MATRIX_EXTRACT_FLAG = 1
save_dict_name_scores = 'food724_scores'
save_dict_name_matrix = 'food724_matrix'

save_dict_name_scores_cat = 'food724_scores_cat'

util_file_root = './'
gen_file_dir_name = 'gen'
gen_file_dir = os.path.join(util_file_root, gen_file_dir_name)

# READ PY_CAFFE LOG FILE
log_file = 'logfile.log'
# log_file = "wh_model_food724_aug_poplist_log.log"
with open(os.path.join(util_file_root, log_file), 'r') as log_file:
    log = log_file.read()

pop_set = set()
with open('pop_list.txt','r') as f:
    lines = f.readlines()
    for line in lines:
        pop_set.add(line.strip('\n'))


# READ LABEL DICT
label_file = os.path.join(util_file_root,'food_id.txt')
label_dict = {}
with open(label_file,'r') as fp:
    for line in fp:
        b = line.strip('\n').split(':')
        label_dict[int(b[0])] = b[1]
print('label dictionary: ')
print(label_dict)

# READ l2v/v2c DICT
l2v_dict = np.load(os.path.join(util_file_root,'l2v_dict.npy')).item()
v2ci_dict = np.load(os.path.join(util_file_root,'v2ci_dict.npy')).item()
v2c_dict = {}
for key in v2ci_dict:
    v2c_dict[key] = v2ci_dict[key][0]

# SAVE CONFUSION MATRIX: row == [gt_label:[plabel]]
matrix={}
confusion_matrix = np.zeros((NUM_CAT,NUM_CAT))
pattern = r"\((?P<label_r>\d+) , (?P<label_p>\d+)\) \| (?P<count>\d+)"
for r in re.findall(pattern, log):
    confusion_matrix[int(r[0])][int(r[1])] = int(r[2])
    # RECORD THOSE MIS CLASSIFIED INTO OTHER CLASS (COUNT > DISPLAY_TH)
    DISPLAY_TH = 1
    if int(r[0]) not in matrix:
        if int(r[2]) > DISPLAY_TH:
            matrix[int(r[0])]=[int(r[0]),[int(r[1]),int(r[2])]]
        else:
            matrix[int(r[0])]=[int(r[0])]
    else:
        if int(r[2]) > DISPLAY_TH:
            matrix[int(r[0])].append([int(r[1]),int(r[2])])
np.save(os.path.join(gen_file_dir,'confusion_matrix.npy'),confusion_matrix)
print('confusion matrix:')
print(confusion_matrix)
print('confusion matrix shape:')
print(np.shape(confusion_matrix))


if ACCURACY_EXTRACT_FLAG == 1:
    #log='Accuracy[722][top1]: 80.8%\nAccuracy[722][top5]: 92.9%\n124124\n4124124\n1313\nAccuracy[723][top1]: 30.8%\nAccuracy[723][top5]: 22.9%\n'
    pattern = r"Accuracy\[(?P<label>\d+)\]\[top1\]: (?P<accuracy1>\d+\.\d+)\%\nAccuracy.*: (?P<accuracy5>\d+\.\d+)\%"
    accuracy_t1 = {}
    accuracy_t5 = {}
    tb=[]
    # EXTRACT TOP1/TOP5 ACCURACY FROM LOGFILE
    for r in re.findall(pattern, log):
        # print(r)
        accuracy_t1[int(r[0])] = float(r[1])
        accuracy_t5[int(r[0])] = float(r[2])
        tb.append([int(r[0]),float(r[1]),float(r[2])])

    # RECALL:
    recall_dict = {}
    for row in range(0,np.shape(confusion_matrix)[0]):
        cls_cnt = 0
        for col in range(0,np.shape(confusion_matrix)[1]):
            cls_cnt += confusion_matrix[row][col]
        tp = confusion_matrix[row][row]
        recall_dict[row] = 100.*tp/cls_cnt

    # PRECISION:
    precision_dict = {}
    for col in range(0,np.shape(confusion_matrix)[1]):
        cls_cnt = 0
        for row in range(0,np.shape(confusion_matrix)[0]):
            cls_cnt += confusion_matrix[row][col]
        tp = confusion_matrix[col][col]
        precision_dict[col] = 100.*tp/cls_cnt
        # tb.append([col,precision_dict[col],0])

    print('recall_dict')
    print(recall_dict)
    print('precision_dict')
    print(precision_dict)

    # VALIDATE ACCURACY == RECALL
    print('======================= validation =======================')
    for item in tb:
        if recall_dict[int(item[0])]!= float(item[1]):
            # print('accuracy/recall not equal@ %d'%int(item[0]))
            print('%.1f : %.1f'%(recall_dict[int(item[0])], float(item[1])))

    # SCORE BY CLASS
    # [precision, recall@t1, recall@t5]
    scores={}
    #tb for sorting purpose
    tb=[]
    for key in recall_dict:
        scores[label_dict[key]] = [precision_dict[key], recall_dict[key], accuracy_t5[key]]
        tb.append([key, precision_dict[key], recall_dict[key], accuracy_t5[key]])
    print('scores dictionary: ')
    print(scores)

    # SCORE BY CATEGORY DICTIONARY
    # [precision, recall@t1, recall@t5]
    scores_cate={}
    for item in scores:
        if v2c_dict[l2v_dict[item]] not in scores_cate:
            scores_cate[v2c_dict[l2v_dict[item]]] = [[scores[item][0],scores[item][1],scores[item][2]]]
        else:
            scores_cate[v2c_dict[l2v_dict[item]]].append([scores[item][0],scores[item][1],scores[item][2]])
    print('scores_cate dictionary: ')
    print(scores_cate)

    # AVERAGE SCORE BY CATEGORY DICTIONARY
    # [precision, recall@t1, recall@t5]
    for item in scores_cate:
        scores_cate[item]=[np.mean(scores_cate[item],axis=0)[0],\
                           np.mean(scores_cate[item],axis=0)[1],\
                           np.mean(scores_cate[item],axis=0)[2]]
    print('final scores_cate dictionary: ')
    print(scores_cate)

    # SCORE BY CATEGORY LIST
    # [precision, recall@t1, recall@t5]
    tb_cate=[]
    for item in scores_cate:
        tb_cate.append([item,scores_cate[item][0],scores_cate[item][1],scores_cate[item][2]])
    print('scores_cate table: ')
    print(tb_cate)
    tb_cate_sorted_t1 = sorted(tb_cate, key=operator.itemgetter(2), reverse=False)
    tb_cate_sorted_t5 = sorted(tb_cate, key=operator.itemgetter(3), reverse=False)

    #SORTING
    tb_sorted_t1 = sorted(tb, key=operator.itemgetter(2), reverse=False)
    tb_sorted_t5 = sorted(tb, key=operator.itemgetter(3), reverse=False)
    print('tb_sorted_t1')
    print(tb_sorted_t1)
    print('tb_sorted_t5')
    print(tb_sorted_t5)

    # tb = tb_sorted_t5
    #np.save(save_dict_name_scores+'.npy',tb)

    metric_name = 'Accuracy'
    # metric_name = 'Recall'

    # [cnt,a1,a2,a3]
    cnt = np.array([0.,0.,0.,0.])
    # WRITE CLASS SCORE FILE
    os.path.join(gen_file_dir,'confusion_matrix.npy')
    with open(os.path.join(gen_file_dir,save_dict_name_scores+'.txt'),'w') as fp:
        s = '{0:50} :  {1:>5}  {2:>5}  {3:>5}\n'.format('Visual Food Label', 'Precision', metric_name, metric_name+'@t5')
        fp.write(s)
        fp.write('----------------------------------------------------------------------------------------\n')
        for item in tb_sorted_t1:
            # if label_dict[item[0]] in pop_set:
            #     continue
            cnt += np.array([1., item[1], item[2], item[3]])
            s = '{0:50} :  {1:>5.1f}  {2:>5.1f}  {3:>5.1f}\n'.format(label_dict[item[0]], item[1], item[2], item[3])
            fp.write(s)
            fp.write('----------------------------------------------------------------------------------------\n')
        s = '{0:50} :  {1:>5.1f}  {2:>5.1f}  {3:>5.1f}\n'.format('Average', cnt[1]/cnt[0],cnt[2]/cnt[0],cnt[3]/cnt[0])
        fp.write(s)
    # for item in tb:
    #     if item[2]<20:
    #         print(item)
    ''''''
    # WRITE CATEGORY SCORE FILE
    with open(os.path.join(gen_file_dir,save_dict_name_scores_cat+'.txt'),'w') as fp:
        s = '{0:50} :  {1:>5}  {2:>5}  {3:>5}\n'.format('Visual Category Name', 'Precision', metric_name, metric_name+'@t5')
        fp.write(s)
        fp.write('----------------------------------------------------------------------------------------\n')
        for item in tb_cate_sorted_t1:
            s = '{0:50} :  {1:>5.1f}  {2:>5.1f}  {3:>5.1f}\n'.format(item[0], item[1], item[2], item[3])
            fp.write(s)
            fp.write('----------------------------------------------------------------------------------------\n')
    ''''''

cnt=0
''''''
if CONFUSION_MATRIX_EXTRACT_FLAG == 1:

    # WRITE CONFUSION MATRIX FILE
    with open(os.path.join(gen_file_dir,save_dict_name_matrix+'.txt'),'w') as fp:
        fp.write('----------------------------------------------------------------------------------------\n')
        fp.write('Visual Food Label: | {0:>6s} | {1:>6s} | {2:>6s} |\n'.format('Precision', metric_name, metric_name+'@t5'))
        rk=0
        #for lb in matrix:
        for entry in tb_sorted_t1:
            rk+=1
            lb=entry[0]
            idx = matrix[lb][0]
            item = matrix[lb][1:]
            # if label_dict[idx] not in pop_set:
            #     continue
            #print(item)
            fp.write('----------------------------------------------------------------------------------------\n')
            fp.write(str(rk)+'th: ' + label_dict[idx] + \
                     ': | {0:>5.1f} | {1:>5.1f} | {2:>5.1f} |\n'.format(scores[label_dict[idx]][0],scores[label_dict[idx]][1],scores[label_dict[idx]][2]))
            fp.write('----------------------------------------------------------------------------------------\n')
            item = sorted(item, key=operator.itemgetter(1), reverse=True)
            for cnt in range(0,len(item)):
                if label_dict[item[cnt][0]] != label_dict[idx]:
                    s = '\t{0:3d} : {1:50}\n'.format(item[cnt][1], label_dict[item[cnt][0]])
                else:
                    s = '\t*{0:2d} : {1:50}\n'.format(item[cnt][1], label_dict[item[cnt][0]])
                fp.write(s)
''''''

















