
import numpy as np
import re
import operator

NUM_CAT = 724

ACCURACY_EXTRACT_FLAG = 1
CONFUSION_MATRIX_EXTRACT_FLAG = 1
save_dict_name_scores = 'food724_scores'
save_dict_name_matrix = 'food724_matrix'

save_dict_name_scores_cat = 'food724_scores_cat'

label_file = 'C:/Users/hengzhao/Desktop/Learn/CV/Model/trained_model/model_food724_sep/labels.txt'
label_dict = {}
with open(label_file,'r') as fp:
    for line in fp:
        b = line.strip('\n').split(':')
        #print(b)
        label_dict[int(b[0])] = b[1]
print('label dictionary: ')
print(label_dict)

l2v_dict_file = 'C:/Users/hengzhao/Desktop/Learn/CV/Model/trained_model/model_food724_sep/l2v_dict.npy'
l2v_dict=np.load(l2v_dict_file).item()
v2ci_dict_file = 'C:/Users/hengzhao/Desktop/Learn/CV/Model/trained_model/model_food724_sep/v2ci_dict.npy'
v2ci_dict=np.load(v2ci_dict_file).item()
v2c_dict = {}
for key in v2ci_dict:
    v2c_dict[key] = v2ci_dict[key][0]
#scores=np.zeros((NUM_CAT,3))
#matrix=np.zeros((NUM_CAT,NUM_CAT))
''''''
log_file = "food724_sep_conf_log.log"
with open(log_file, 'r') as log_file:
    log = log_file.read()
''''''

tt=[]
#FALSE classify into non_food
ptn = r"expected (?P<label>\d+) but predicted 441"
for r in re.findall(ptn, log):
    tt.append(r[0])
    #print(len(r))

print(tt)
print(len(tt))
scores={}
matrix={}
if ACCURACY_EXTRACT_FLAG == 1:
    #log='Accuracy[722][top1]: 80.8%\nAccuracy[722][top5]: 92.9%\n124124\n4124124\n1313\nAccuracy[723][top1]: 30.8%\nAccuracy[723][top5]: 22.9%\n'
    pattern = r"Accuracy\[(?P<label>\d+)\]\[top1\]: (?P<accuracy1>\d+\.\d+)\%\nAccuracy.*: (?P<accuracy5>\d+\.\d+)\%"
    label_idx = []
    accuracy_t1 = []
    accuracy_t5 = []
    tb=[]
    for r in re.findall(pattern, log):
        # print(r)
        label_idx.append(int(r[0]))
        accuracy_t1.append(float(r[1]))
        accuracy_t5.append(float(r[2]))
        tb.append([int(r[0]),float(r[1]),float(r[2])])


    label_idx = np.array(label_idx)
    accuracy_t1 = np.array(accuracy_t1)
    accuracy_t5 = np.array(accuracy_t5)

    #SAVE_DICT
    for item in tb:
        #scores[item[0]] = [item[1],item[2]]
        scores[label_dict[item[0]]] = [item[1],item[2]]
    print('scores dictionary: ')
    print(scores)

    # score by category dictionary
    scores_cate={}
    for item in scores:
        if v2c_dict[l2v_dict[item]] not in scores_cate:
            scores_cate[v2c_dict[l2v_dict[item]]] = [[scores[item][0],scores[item][1]]]
        else:
            scores_cate[v2c_dict[l2v_dict[item]]].append([scores[item][0],scores[item][1]])
    print('scores_cate dictionary: ')
    print(scores_cate)
    for item in scores_cate:
        scores_cate[item]=[np.mean(scores_cate[item],axis=0)[0],np.mean(scores_cate[item],axis=0)[1]]
    print('final scores_cate dictionary: ')
    print(scores_cate)

    # score by category list
    tb_cate=[]
    for item in scores_cate:
        tb_cate.append([item,scores_cate[item][0],scores_cate[item][1]])
    print('scores_cate table: ')
    print(tb_cate)
    tb_cate_sorted_t1 = sorted(tb_cate, key=operator.itemgetter(1), reverse=False)
    tb_cate_sorted_t5 = sorted(tb_cate, key=operator.itemgetter(2), reverse=False)

    #SORTING
    tb_sorted_t1 = sorted(tb, key=operator.itemgetter(1), reverse=False)
    tb_sorted_t5 = sorted(tb, key=operator.itemgetter(2), reverse=False)
    print('tb_sorted_t1')
    print(tb_sorted_t1)
    print('tb_sorted_t5')
    print(tb_sorted_t5)

    tb = tb_sorted_t5
    #np.save(save_dict_name_scores+'.npy',tb)

    with open(save_dict_name_scores+'.txt','w') as fp:
        s = '{0:50} :  {1:>5}  {2:>5}\n'.format('Visual Food Label', 'Top1', 'Top5')
        fp.write(s)
        fp.write('--------------------------------------------------------------------\n')
        for item in tb:
            s = '{0:50} :  {1:>5.1f}  {2:>5.1f}\n'.format(label_dict[item[0]], item[1], item[2])
            fp.write(s)
            fp.write('--------------------------------------------------------------------\n')
            #fp.write(label_dict[item[0]] + ' : ' + str(item[1]) +', \t'+ str(item[2]) + '\n')
    for item in tb:
        if item[2]<20:
            print(item)
    ''''''
    with open(save_dict_name_scores_cat+'.txt','w') as fp:
        s = '{0:50} :  {1:>5}  {2:>5}\n'.format('Visual Category Name', 'Top1', 'Top5')
        fp.write(s)
        fp.write('--------------------------------------------------------------------\n')
        for item in tb_cate_sorted_t5:
            s = '{0:50} :  {1:>5.1f}  {2:>5.1f}\n'.format(item[0], item[1], item[2])
            fp.write(s)
            fp.write('--------------------------------------------------------------------\n')
            #fp.write(label_dict[item[0]] + ' : ' + str(item[1]) +', \t'+ str(item[2]) + '\n')
    ''''''

cnt=0
if CONFUSION_MATRIX_EXTRACT_FLAG == 1:

    confusion_matrix = np.zeros((NUM_CAT,NUM_CAT))
    tt=0
    pattern = r"\((?P<label_r>\d+) , (?P<label_p>\d+)\) \| (?P<count>\d+)"
    for r in re.findall(pattern, log):
        if tt==0:
            print(r)
            tt=-1
        confusion_matrix[int(r[0])][int(r[1])] = int(r[2])
        if int(r[0]) not in matrix:

            #if int(r[2]) !=0:
            if int(r[2]) > 1:
                matrix[int(r[0])]=[int(r[0]),[int(r[1]),int(r[2])]]
            else:
                matrix[int(r[0])]=[int(r[0])]
            #matrix[int(r[0])]=[[int(r[1]),int(r[2])]]
        else:
            #if int(r[2]) !=0:#> 2:
            if int(r[2]) > 1:
                '''
                if cnt <10:
                    cnt +=1
                    print(r)
                '''
                matrix[int(r[0])].append([int(r[1]),int(r[2])])
    print(confusion_matrix)
    np.save('confusion_matrix.npy',confusion_matrix)
    #print(matrix[0])
    #print(matrix[1])
    #print(matrix[2])
    #print(tb_sorted_t1)
    ''''''
    with open(save_dict_name_matrix+'.txt','w') as fp:
        #print(matrix)
        fp.write('--------------------------------------------------------------------\n')
        fp.write('Visual Food Label: | {0:>6s} | {1:>6s}\n'.format('Top 1','Top 5'))
        rk=0
        #for lb in matrix:
        for entry in tb_sorted_t5:
            rk+=1
            lb=entry[0]
            idx = matrix[lb][0]
            item = matrix[lb][1:]
            #print(item)
            fp.write('--------------------------------------------------------------------\n')
            fp.write(str(rk)+'th: ' + label_dict[idx] + \
                     ': | {0:>5.1f} | {1:>5.1f} |\n'.format(scores[label_dict[idx]][0],scores[label_dict[idx]][1]))
            fp.write('--------------------------------------------------------------------\n')
            item = sorted(item, key=operator.itemgetter(1), reverse=True)
            for cnt in range(0,len(item)):
                #s = '\t{0:>50}  :  {1:>2d}\n'.format(label_dict[item[cnt][0]],item[cnt][1])
                if label_dict[item[cnt][0]] != label_dict[idx]:
                    s = '\t{0:3d} : {1:50}\n'.format(item[cnt][1], label_dict[item[cnt][0]])
                else:
                    s = '\t*{0:2d} : {1:50}\n'.format(item[cnt][1], label_dict[item[cnt][0]])
                fp.write(s)
                #fp.write('--------------------------------------------------------------------\n')
    ''''''

















