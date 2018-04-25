# caffe_confusion_matrix

Calculate confusion matrix in pycaffe

# useage

pycaffe_confmat.py for build the log file

cmd like this one:

```shell
python pycaffe_confmat.py --proto model\ResNet-50-deploy.prototxt --model model\model_93000.caffemodel --lmdb D:\dataset\newFood_724_clean\images\newFood724_test_lmdb --mean model\newFood724_aug_mean.binaryproto
```

extract_log.py for building confusion matrix
extract_log_most_conf.py for building most confusing matrix