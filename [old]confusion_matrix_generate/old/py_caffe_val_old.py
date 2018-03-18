import sys
import caffe
import matplotlib
import numpy as np
import lmdb
import argparse
from collections import defaultdict
from PIL import Image
import logging

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
    correct = 0
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
    for key, value in lmdb_cursor:
	cnt+=1
	datum = caffe.proto.caffe_pb2.Datum()
	datum.ParseFromString(value)
	label = int(datum.label)
	image = caffe.io.datum_to_array(datum)
	image = image.astype(np.uint8)
	#print(image.shape)
	image = image[:,16:240,16:240]
	#print((np.asarray([image])).shape)
        #print(mean_image.shape)
	
	out = net.forward_all(data=np.asarray([image])-mean_image)
	#out = net.forward()
	#out = net.forward_all(data=np.asarray([image]))
	plabel = int(out['prob'][0].argmax(axis=0))
	#print(out[:][0][0:5].argmax(axis=0))
	
        plabel_all = np.argsort(out['prob'][0])[::-1]
	plabel_top5 = plabel_all[0:5]

        count = count + 1
        iscorrect = label == plabel
        correct = correct + (1 if iscorrect else 0)
        matrix[(label, plabel)] += 1
        labels_set.update([label, plabel])

        if not iscorrect:
            print("Error: key=%s, expected %i but predicted %i" \
                    % (key, label, plabel))
	
	# fix: assuming no shuffle in test_lmdb
	if cnt==1:
	    tmp = label	
	# assuming first label is 0 and no shuffle in test_lmdb:  WRONG!
	print(tmp)
	print(label)
	break
        if tmp == label:
            count_tmp += 1
            if label in plabel_top5:
                correct_top5 += 1
	    if label == plabel:
                correct_top1 += 1
        else:
            print("Accuracy[%d][top1]: %.1f%%" % (label-1,100.*correct_top1/count_tmp))
            print("Accuracy[%d][top5]: %.1f%%" % (label-1,100.*correct_top5/count_tmp))
            tmp = label
            count_tmp = 1
	    correct_top1 = 0
	    correct_top5 = 0
	#if cnt==200:
	    #break
        #sys.stdout.write("\rAccuracy: %.1f%%" % (100.*correct/count))
        #sys.stdout.flush()

    print("Accuracy[%d][top1]: %.1f%%" % (label,100.*correct_top1/count_tmp))
    print("Accuracy[%d][top5]: %.1f%%" % (label,100.*correct_top5/count_tmp))
    #print('total number of test images: %d',%cnt)
    print(cnt)

    print('\n' + str(correct) + " out of " + str(count) + " were classified correctly")

    print("")
    print("Confusion matrix:")
    print("(r , p) | count")
    for l in labels_set:
        for pl in labels_set:
            print("(%i , %i) | %i" %(l, pl, matrix[(l,pl)]) )


