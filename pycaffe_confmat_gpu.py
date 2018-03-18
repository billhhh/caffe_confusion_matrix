#
#Author: Bill Wang
#

import sys
import caffe
import matplotlib
import numpy as np
import lmdb
import argparse
from collections import defaultdict
import os

TEST_FROM_IMAGE_FILE = False
from PIL import Image
import matplotlib

import logging

# IMAGE_FILE = './macaron.jpeg'
# IMAGE_FILE = './tea-600x400.jpg'
IMAGE_FILE = './orange-web-1024x768.jpg'

test_root = '../newFood_724_clean/images'
test_txt = 'newFood724_test.txt'
label_txt = 'food_id.txt'

test_files = []
with open(test_txt,'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        for i in range(0,5):
            if line[-i] == ' ':
                idx = i
                break
        test_files.append([os.path.join(test_root,line[:-i]), int(line[-i:])])

label_dict = {}
with open(label_txt,'r') as fp:
    for line in fp:
        b = line.strip('\n').split(':')
        #print(b)
        label_dict[int(b[0])] = b[1]

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--lmdb', type=str, required=True)
    parser.add_argument('--mean', type=str, required=True)
    args = parser.parse_args()
    
    sys.stdout = Logger()
    '''
    sys.stdout = Logger()
    #logging
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler("{0}/{1}.log".format('.', 'full_test_food724'))
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    rootLogger.addHandler(consoleHandler)
    '''
    count = 0
    count_tmp = 1 
    correct_top1 = 0
    correct_top5 = 0
    matrix = defaultdict(int) # (real,pred) -> int
    labels_set = set()

    mean_file_binaryproto = args.mean

    # Extract mean from the mean image file
    mean_blobproto_new = caffe.proto.caffe_pb2.BlobProto()
    with open(mean_file_binaryproto, 'rb') as f:
        mean_blobproto_new.ParseFromString(f.read())
        mean_image = caffe.io.blobproto_to_array(mean_blobproto_new)

    #mean_image_new = mean_image[0:2][16:240][16:240]
    mean_image_orig = mean_image
    mean_image = mean_image[:,:,16:240,16:240]
    #print(mean_image.shape)

    net = caffe.Net(args.proto, args.model, caffe.TEST)
    classifier = caffe.Classifier( \
        mean=mean_image[0].mean(1).mean(1), \
        model_file=args.proto, \
        pretrained_file=args.model, \
        channel_swap=(2,1,0), \
        raw_scale=255, \
        image_dims=(256,256)
    )

    caffe.set_mode_gpu()
    #caffe.set_mode_cpu()
    lmdb_env = lmdb.open(args.lmdb)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    #Define image transformers
    #transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    #transformer.set_mean('data', mean_array)
    #transformer.set_transpose('data', (2,0,1))
    #lmdb_cursor=lmdb_cursor[:10]
    print('============================')
    print(mean_image_orig.shape)
    print(mean_image.shape)
    scores_top1=[]
    scores_top5=[]
    tmp=0
    cnt=0
    my_dict={}
    lb_set=set()
    if TEST_FROM_IMAGE_FILE == True:
        for fn, label in test_files:
            cnt+=1
            # if cnt < 4:
            #     continue
            IMAGE_FILE = fn
            try:
                inputs = caffe.io.load_image(IMAGE_FILE)
            except Exception as e:
                print(e.message)
                continue
            # inputs = caffe.io.load_image('tt.jpeg')
            # print('input shape')
            # print(inputs.shape)

            res = classifier.predict(inputs=np.asarray([inputs]), oversample=True)
            plabel = res[0].argmax(axis=0)
            plabel_top5 = np.argsort(res[0])[::-1][0:5]

            # print([label_dict[item] for item in plabel_top5])
            # print('p:  %s | gt: %s'%(label_dict[plabel],label_dict[label]))

            count = count + 1
            iscorrect = (label == plabel)
            correct_top1 = correct_top1 + (1 if iscorrect else 0)
            iscorrect_t5 = (label in plabel_top5)
            correct_top5 = correct_top5 + (1 if iscorrect_t5 else 0)

            matrix[(label, plabel)] += 1
            labels_set.update([label, plabel])

            if not iscorrect:
                print("Error: key=%s, expected %i but predicted %i" \
                        % (cnt, label, plabel))

            lb_set.add(label)
            if label not in my_dict:
                #count: [top1, top5, num]
                my_dict[label] = [0,0,0]
            #num increase
            my_dict[label] = [my_dict[label][0], my_dict[label][1], my_dict[label][2]+1]

            if label == plabel:
                my_dict[label] = [my_dict[label][0]+1,my_dict[label][1],my_dict[label][2]]

            if label in plabel_top5:
                my_dict[label] = [my_dict[label][0],my_dict[label][1]+1,my_dict[label][2]]

            # break
            # if cnt > 10:
            #     break
    else:
        for key, value in lmdb_cursor:
            cnt+=1
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)
            label = int(datum.label)
            image = caffe.io.datum_to_array(datum)
            image = image.astype(np.uint8)
            # CENTER CROP
            image = image[:,16:240,16:240]

            out = net.forward_all(data=np.asarray([image])-mean_image)
            #out = net.forward()
            #out = net.forward_all(data=np.asarray([image]))
            plabel = int(out['prob'][0].argmax(axis=0))
            #print(out[:][0][0:5].argmax(axis=0))

            plabel_all = np.argsort(out['prob'][0])[::-1]
            plabel_top5 = plabel_all[0:5]

            count = count + 1
            iscorrect = label == plabel
            correct_top1 = correct_top1 + (1 if iscorrect else 0)
            iscorrect_t5 = (label in plabel_top5)
            correct_top5 = correct_top5 + (1 if iscorrect_t5 else 0)

            matrix[(label, plabel)] += 1
            labels_set.update([label, plabel])

            if not iscorrect:
                print("Error: key=%s, expected %i but predicted %i" \
                        % (key, label, plabel))

            lb_set.add(label)
            if label not in my_dict:
                #count: [top1, top5, num]
                my_dict[label] = [0,0,0]
            #num increase
            my_dict[label] = [my_dict[label][0], my_dict[label][1], my_dict[label][2]+1]

            if label == plabel:
                my_dict[label] = [my_dict[label][0]+1,my_dict[label][1],my_dict[label][2]]

            if label in plabel_top5:
                my_dict[label] = [my_dict[label][0],my_dict[label][1]+1,my_dict[label][2]]
            #if cnt==200:
                #break
                #sys.stdout.write("\rAccuracy: %.1f%%" % (100.*correct/count))
                #sys.stdout.flush()

    # FOR LOG EXTRACTION

    # PRINT CLASS ACCURACY
    for label in my_dict:
        print("Accuracy[%d][top1]: %.1f%%" % (label,100.*my_dict[label][0]/my_dict[label][2]))
        print("Accuracy[%d][top5]: %.1f%%" % (label,100.*my_dict[label][1]/my_dict[label][2]))
    np.save('score_t1_t5.npy',my_dict)
    # PRINT OVERAL ACCURACY
    print('TOP1\n' + str(correct_top1) + " out of " + str(count) + " were classified correctly")
    print('TOP5\n' + str(correct_top5) + " out of " + str(count) + " were classified correctly")

    # CONFUSION MATRIX
    print("")
    print("Confusion matrix:")
    print("(r , p) | count")
    for l in labels_set:
        for pl in labels_set:
            print("(%i , %i) | %i" %(l, pl, matrix[(l,pl)]) )
''''''

