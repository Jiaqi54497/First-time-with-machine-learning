# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:34:18 2020

@author: Lenovo
"""
import numpy as np
"""SST-2"""

def read_sentences():
    """将每一行读取成一个由单词组成的list"""
    txtpath = 'D:/Machine_learning/SST-2/original/SOStr.txt'
    fpp = open(txtpath)
    all_sentences = fpp.readlines()
    for i in range(len(all_sentences)):
        all_sentences[i] = all_sentences[i].replace('\n','').split('|')
    return all_sentences

all_sentences = read_sentences()
#print(all_sentences[0])

#%%
import pandas as pd
path_dev = 'D:/Machine_learning/SST-2/dev.tsv'
path_train = 'D:/Machine_learning/SST-2/train.tsv'
path_test = 'D:/Machine_learning/SST-2/test.tsv'

def read_tsv(path):
    file = pd.read_csv(path,sep='\t')
    dict_file = file.to_dict()
    print("Have read the file as a dictionary.")
    return dict_file

dev_dic = read_tsv(path_dev)
train_dic = read_tsv(path_train)
#test_dic = read_tsv(path_test)
#%%
def read_sentences(dic):
    lis = []
    for i in range(len(dic['sentence'])):
        lis.append(dic['sentence'][i])
    print("Return a sentence list with %d elements."%(len(lis)))
    return lis

dev_list = read_sentences(dev_dic)
train_list = read_sentences(train_dic) 
#test_list = read_sentences(test_dic)

#%%
def split_sentences(s):
    for i in range(len(s)):
        s[i] = s[i].replace('.','').replace(',','').split(' ')
    print("Have splited %d sentences."%(len(s)))
    return s

dev_split = split_sentences(dev_list)
train_split = split_sentences(train_list)
#test_split = split_sentences(test_list)
#%%
def create_my_dictionary(all_sentences):
    """创建包含所有词的字典，顺便将每个句子化成向量"""
    dictionary = ['The']
    for i in range(len(all_sentences)):
        for j in range(len(all_sentences[i])):
            c = 0
            for k in range(len(dictionary)):
                if all_sentences[i][j] == dictionary[k]:
                    break
                if all_sentences[i][j] != dictionary[k]:
                    c += 1
            if c == len(dictionary):
                dictionary.append(all_sentences[i][j])
        print("Checking words in the %dth sentence. ---Done"%(i))
    print("Have created a dictionary with %d words."%(len(dictionary)))
    return dictionary
    
dictionary = create_my_dictionary(train_split)
#%%
def look_up_dic(dic,sp):
    """查字典，把句子化成向量"""   
    ar = np.empty((len(sp),len(dic)),dtype = np.float16)
    for i in range(len(sp)):
        arr = np.zeros(len(dic))
        for j in range(len(sp[i])):
            k = 0
            for k in range(len(dic)):
                if sp[i][j] == dic[k]:
                    arr[k] += 1
        ar[i] = arr
        print("Have looked up the %dth sentence."%(i))
    print("All %d sentences done."%(len(sp)))
    return ar

dev_vector = look_up_dic(dictionary, dev_split)
train_vector = look_up_dic(dictionary, train_split)
#test_vector = look_up_dic(dictionary, test_split)

#%%
def find_labels(dic):
    arr = np.empty(len(dic['label']))
    for i in range(len(dic['label'])):
        arr[i] = dic['label'][i]
    print("Have read %d labels."%(len(arr)))
    return arr

labels_dev = find_labels(dev_dic)
labels_train = find_labels(train_dic)
#print(labels_train[:20])
#labels_test = find_labels(test_dic)        
#%%
def normalize(x):
    x = np.true_divide(x,x.max(),)
    x = np.concatenate((x,np.ones((x.shape[0],1))),axis=1)
    print("The array has been normalized and added one external column. The shape is now",x.shape)
    return x
#%%
dev = normalize(dev_vector)
train = normalize(train_vector)
#%%

#%%
"""Linear Model"""
import random
def predict_labels(x,w):
    y = fwx(x,w)
    for i in range(len(y)):
        if y[i] >= 0:
            y[i] = 1
        if y[i] < 0:
            y[i] = 0
    print("Have predicted %d labels."%(len(y)))
    return y
#%%
def predict_multilabels(x,w):
    y = fwx(x,w)
    #print(y[0].shape)
    labels = np.zeros(len(y))
    for i in range(len(y)):
        labels[i] = np.argmax(y[i])
    #print(labels[:10])
    #print("Have predicted multilabels with shape",labels.shape)
    return labels
#%%
def lower_dim_y(y):
    low_y = np.zeros(len(y))
    for i in range(len(low_y)):
        low_y[i] = np.argmax(y[i])
    return low_y
#%%
#print(train_labels_mn_max.shape)
#print(lower_dim_y(train_labels_mn_max)[:10])
#print(train_labels_mnist[:10])
#%%
def fwx(x,w):
    #print(x.shape,w.shape)
    f = 1/(1+np.exp(-np.dot(x,w)))
    return f
#%%
def log_likelihood(x,y,w):
    l = 0
    for i in range(len(y)):
        f = fwx(x[i],w)
        l += y[i]*np.log(f)+(1-y[i])*np.log(1-f)
        #print(y[i]*np.log(fwx(x,w)[i]),fwx(x,w))
    return l

def train_w(x,y,a,k):
    """Minimize the training error"""
    w = np.random.random((x.shape[1],))
    l_new = l_old = 1
    c = 1
    while l_new <= l_old:
        l_old = l_new
        #i2 = 0
        #group = []
#        for i2 in range(s):            
        for i in range(k):
            t = random.randint(0, len(x)-1)
            #print(t)
            w += a*(y[t]-fwx(x[t],w))*(x[t].T)
            #group.append(w)
        print("Have updated w %d times."%(c))
        #l_newg = np.ones(s)
        #for i2 in range(s):            
            #print(len(group)
            #l_newg[i2] = error(x, y, group[i2])
        #print(l_newg)
        l_new = error(x,y,w)        
        #print(l_new)
        c += 1
        
    print("Finished learning. The min error is %f"%(l_new))
    return w
#%%
def train_valid(x_train,x_validation,y_train,y_validation,a,k):
    """Linear model for binary classification."""
    w = np.random.random((x_train.shape[1],))
    l_new = l_old = 1
    c = 1
    while l_new <= l_old:
        l_old = l_new           
        for i in range(k):
            t = random.randint(0, len(x_train)-1)
            #print(t)
            w += a*(y_train[t]-fwx(w.T,x_train[t]))*(x_train[t].T)
        print("Have updated w %d times."%(c))
        l_new = error(x_validation,y_validation,w)        
        #print(l_new)
        c += 1        
    print("Finished learning. The min error on this validation set is %f"%(l_new))
    return w
#%%
def train_valid_softmax2(x_train,x_validation,y_train,y_validation,a,k):
    """Minimizing error on validation set"""
    #print(y_train.shape)
    w0 = w1 = np.random.random((x_train.shape[1],y_train.shape[1]))
    l_new = l_old = 1
    c = 1
    while l_new <= l_old:
        l_old = l_new
        w0 = w1           
        for i in range(k):
            t = random.randint(0, len(x_train)-1)
            #print(y_train[0].shape,x_train[0].T.shape,w.shape,fwx(w.T,x_train[t]).shape)
            w1 = w0 + a*((y_train[t]-fwx(w0.T,x_train[t]))*(x_train[t].reshape(1,len(x_train[t])).T))
        print("Have updated w %d times."%(c))
        l_new = error_multi(x_validation,y_validation,w1)        
        #print(l_new)
        c += 1        
    print("Finished learning. The min error on this validation set is %f"%(l_old))
    return w0

#%%
def k_fold(x,y,a,k,part):
    """"k-fold function for binary linear classification."""
    partition = len(x)//part
    w = []
    for i in range(part):
        x_validation = x[part:part+partition]
        y_validation = y[part:part+partition]
        x_train = np.concatenate((x[:part],x[(part+partition):]),axis = 0)
        y_train = np.concatenate((y[:part],y[(part+partition):]),axis = 0)
        w_k = train_valid(x_train,x_validation,y_train,y_validation,a,k)
        w.append(w_k)
        print("Finished the %dth fold validation."%(i))
    w_all = np.zeros(w[0].shape)
    for i in range(len(w)):
        w_all += w[i]
    w_ave = w_all/part
    print("Function k_fold: done.")
    return w_ave
#%%
def k_fold_multi(x,y,a,k,part,th):
    """"k-fold function for multiple-class linear classification."""
    #print(y.shape)
    y_max = softmax_y(y)
    partition = len(x)//part
    w = []
    for i in range(part):
        x_validation = x[part:part+partition]
        y_validation = y_max[part:part+partition]
        x_train = np.concatenate((x[:part],x[(part+partition):]),axis = 0)
        y_train = np.concatenate((y_max[:part],y_max[(part+partition):]),axis = 0)
        #print(y_train.shape,y_max)
        w_k = train_valid_softmax(x_train,x_validation,y_train,y_validation,a,k,th)
        w.append(w_k)
        print("Finished the %dth fold_multi validation."%(i))
    w_all = np.zeros(w[0].shape)
    for i in range(len(w)):
        w_all += w[i]
        #print("w_all:",w_all)
    w_ave = w_all/part
    #print("w_ave:",w_ave)
    er_ave = error_multi(x, y_max, w_ave)
    print("Function k_fold_multi: done. Error on the whole training set is %f"%(er_ave))
    return w_ave

#%%
#k_fold_multi(x_test_mnist,y_test_mnist,1,5000,2,0.01)
#w_ave = k_fold(train[:1000],labels_train[:1000],0.01,100,10) 
#%%   
#w = train_w(train,labels_train,0.01,100)  
#%%
def error(x,y,w):
    l = 0
    p = predict_labels(x, w)
    for i in range(len(y)):       
        #print(y.shape,p.shape)
        if y[i] != p[i]:
            l += 1
    r = l/len(y)
    print("The error rate is %f"%(r))
    return r
#%%
def error_multi(x,y,w):
    l = 0
    p = predict_multilabels(x, w)
    y = lower_dim_y(y)
    #print(y[:10],p[:10])
    for i in range(len(y)):              
        if y[i] != p[i]:
            l += 1
    r = l/len(y)
    print("The error rate is %f"%(r))
    return r
   #%% 
#train_error = error(train,labels_train,w)
#%%
"""MINIST"""
import struct
import matplotlib.pyplot as plt

train_images_idx3_ubyte_file = 'D:/Machine_learning/train-images-idx3-ubyte/train-images.idx3-ubyte'
train_labels_idx1_ubyte_file = 'D:/Machine_learning/train-labels-idx1-ubyte/train-labels.idx1-ubyte'
test_images_idx3_ubyte_file = 'D:/Machine_learning/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte'
test_labels_idx1_ubyte_file = 'D:/Machine_learning/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte'

def decode_idx3_ubyte(idx3_ubyte_file):
    
# 打开一个文件并进行读取操作，读取的内容放在缓冲区中，read()表示全部读取

    bin_data = open(idx3_ubyte_file, 'rb').read()
    
#struct.unpack_from(fmt=,buffer=,offset=)函数可以将缓冲区buffer中的内容在按照指定的格式fmt
#从偏移量为offset=numb的位置开始进行读取。返回的是一个对应的元组tuple

    offset = 0
    
#用来读header的格式
    fmt_header = '>iiii'
    
 # 解析文件头信息，依次为魔数、图片数量、每张图片高（行数）、每张图片宽（列数）

    magic_num, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print ('Read from the idx3 header---\nmagic number:%d, image number: %d, image size: %d*%d' %(magic_num, num_images, num_rows, num_cols))

#解析数据集  
    image_size = num_cols * num_rows
    fmt_image = '>' + str(image_size) + 'B'
     #内容解析格式，其中B表示的是一个字节 > 表示的是大端法则，image_size表示的是多少个字节。
     
    offset += struct.calcsize(fmt_header)
    #返回格式字符串fmt描述的结构的字节大小
    
    #images = np.empty((num_images, num_rows, num_cols))
    images = np.empty((num_images, num_rows*num_cols))
    #empty(shape):根据给定的维度和数值类型返回一个新的数组其元素不进行初始化；shape是整数或者整数组成的元组，代表空数组的维度
    for i in range(num_images):
        #images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset))
        #reshape((a,b))把原数据变成a行b列的格式。image[i]是一个矩阵
        offset += struct.calcsize(fmt_image)
    return images, num_images
    
def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_num, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print("Read from the idx1 header---\nmagic number: %d, number of images: %d" %(magic_num, num_images))
    
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]#这个[0]是什么意思啊？
        offset += struct.calcsize(fmt_image)
    return labels

def load_data(idx1_ubyte_file, idx3_ubyte_file):
    images, num_images = decode_idx3_ubyte(idx3_ubyte_file)
    labels = decode_idx1_ubyte(idx1_ubyte_file)
    print("Successfully loaded %d images and labels" %(num_images))
    return images, labels

def show_image(images, labels, image_number):
    print(labels[image_number])
    plt.imshow(images[image_number].reshape(28,28), cmap = 'gray')
    plt.show()
#%%
train_images_mnist, train_labels_mnist = load_data(train_labels_idx1_ubyte_file, train_images_idx3_ubyte_file)
test_images_mnist, test_labels_mnist = load_data(test_labels_idx1_ubyte_file, test_images_idx3_ubyte_file)
#%%
train_images_mn_nor = normalize(train_images_mnist)
test_images_mn_nor = normalize(test_images_mnist)
#%%
def softmax_y(y):
    max = np.int(y.max())
    y_max = np.zeros((len(y),max+1))
    for i in range(len(y_max)):
        y_max[i][np.int(y[i])]  = 1
    print("Have changed y into an array with shape",y_max.shape)
    return y_max
#%%
#train_labels_mn_max = softmax_y(train_labels_mnist)
#test_labels_mn_max = softmax_y(test_labels_mnist)
#%%
#w_mnist = k_fold_multi(train_images_mn_nor[:1000], train_labels_mnist[:1000], 0.01, 100, 10)
#%%
"""Neural Network"""
def nn_predict(x,v,w):
    h = fwx(x,v)
    #print(h.shape,w.shape)
    y = fwx(h,w)
    labels = lower_dim_y(y)
    #print("Function nn_predict:return labels with shape",labels.shape)
    return labels

#%%

def backpropagation(x_train,x_validation,y_train,y_validation,a,k,hid,threshold):
    """Grediant descent method for feedforward neural networks with one hidden layer."""
    y_train = softmax_y(y_train)
    y_validation = softmax_y(y_validation)
    v = np.random.random((x_train.shape[1],hid)) - np.random.random((x_train.shape[1],hid))
    w = np.random.random((hid,y_train.shape[1])) - np.random.random((hid,y_train.shape[1]))

    l_new = 1
    l_old = 0
    c = 1
    yd = len(y_train[0])
    xd = len(x_train[0])
    #print("abs:",abs(l_new - l_old))
    while abs(l_new - l_old) >= threshold:
        l_old = l_new                    
        for i in range(k): 
            t = random.randint(0, len(x_train)-1)
            alpha = np.dot(x_train[t],v)
            h = 1/(1+np.exp(-alpha))
            pre = fwx(h,w)
            yty = np.dot(pre.reshape(1,yd),1-pre.reshape(yd,1))
            w -= a*yty*np.dot(h.reshape(hid,1),pre.reshape(1,yd)-y_train[t].reshape(1,yd))
            eph = -yty*np.dot(w,(pre.reshape(yd,1)-y_train[t].reshape(yd,1)))
            hth = np.dot(h.reshape(1,hid),1-h.reshape(hid,1))
            v += a*hth*np.dot(x_train[t].reshape(xd,1),eph.reshape(1,hid))
            #print("yty:",yty,"hth:",hth,"h:",h,"pre:",pre,"alpha",alpha)
            #break

        print("Have updated w %d times."%(c))
        l_new = nn_error(x_validation,y_validation,v,w)        
        #print(v,w)
        c += 1        
    print("Function backpropagation ---done. The min error on this validation set is %f"%(l_old))
    print("The error on the whole training set is %f."%(nn_error(x_train, y_train, v, w)))
    return w,v
#%%
#backpropagation(x_train_mnist,x_train_mnist,y_train_mnist,y_train_mnist,0.02,10000,50,0.001)
#%%
def nn_error(x,y,v,w):
    pre = nn_predict(x, v, w)
    yone = lower_dim_y(y)
    error = 0
    #print(pre[:10],yone[:10])
    for i in range(len(pre)):
        if pre[i] != yone[i]:
            error += 1
    #print(error,len(pre))
    er_rate = error/len(pre)
    print("The nn_error rate is %f"%(er_rate))
    return er_rate
#%%
def k_fold_nn(x,y,a,k,hid,part,th):
    """A k_fold function for neural networks."""
    #print(y.shape)
    y_max = softmax_y(y)
    partition = len(x)//part
    w = []
    v = []
    for i in range(part):
        x_validation = x[part:part+partition]
        y_validation = y_max[part:part+partition]
        x_train = np.concatenate((x[:part],x[(part+partition):]),axis = 0)
        y_train = np.concatenate((y_max[:part],y_max[(part+partition):]),axis = 0)
        #print(y_train.shape,y_max)
        w_k,v_k = backpropagation(x_train,x_validation,y_train,y_validation,a,k,hid,th)
        w.append(w_k)
        v.append(v_k)
        print("k_fold_nn: Finished the %dth fold_multi validation."%(i))
        break
    w_all = np.zeros(w[0].shape)
    v_all = np.zeros(v[0].shape)
    for i in range(len(w)):
        w_all += w[i]
        v_all += v[i]
    w_ave = w_all/part
    v_ave = v_all/part
    print("Function k_fold_nn: done. The error rate on the whole training set is %f."%(nn_error(x, y_max, v_all, w_all)))
    return w_ave,v_ave
#%%
#w_mn_nn, v_mn_nn = k_fold_nn(train_images_mn_nor[:1000], train_labels_mnist[:1000],0.01,500,6,10)
#%%
#w_sst_nn, v_sst_nn = k_fold_nn(train[:1000],labels_train[:1000],0.01,100,6,10)
#%%
"""经过整理的变量。这里所有x都经过了归一化，并且加上一列，作为线性组合的常数项。
各变量的维数为：
x_train_sst (67349,14790)
x_test_sst (872,14790)
y_train_sst (67349,)
y_test_sst (872,)
x_train_mnist (60000,785)
x_test_mnist (10000,785)
y_train_mnist (60000,)
y_test_mnist (10000,)
"""
x_train_sst = train
x_test_sst = dev
y_train_sst = labels_train
y_test_sst = labels_dev
x_train_mnist = train_images_mn_nor
x_test_mnist = test_images_mn_nor
y_train_mnist = train_labels_mnist
y_test_mnist = test_labels_mnist
#%%
""""二分类线性模型，对于mnist进行调参。共有三个参数，步长a，mini_batch里面取的集合大小k，以及k-fold中的分块数目part."""
from datetime import datetime
def para_k(x_train,x_test,y_train,y_test,a,part,ran,hid,th):
    """"最后一个参数nn才有"""
    tr = []
    te = []
    ti = []
    for k in ran:
        print("Running the k=%d test."%(k))
        start = datetime.now()
        w = k_fold_multi(x_train, y_train, a, k, part,th)
        train_error = error_multi(x_train,softmax_y(y_train),w)
        test_error = error_multi(x_test,softmax_y(y_test),w)
        #w,v = k_fold_nn(x_train, y_train, a, k,hid, part)
        #train_error = nn_error(x_train,y_train,v,w)
        #test_error = nn_error(x_test,y_test,v,w)
        run_time = datetime.now() - start
        tr.append(train_error)
        te.append(test_error)
        ti.append(run_time.seconds)
        
    plt.plot(ran,tr,label = 'train error')
    plt.plot(ran,te,label = 'test error')
    plt.xlabel('test range of k')
    plt.ylabel('error rate')
    plt.title('train and test error')
    plt.show()
    
    plt.plot(ran,ti)
    plt.xlabel('test range of k')
    plt.ylabel('seconds')
    plt.title('run time')
    plt.show()
#%%
def para_a(x_train,x_test,y_train,y_test,ran,part,k,hid,th):
    """"hid参数nn才有"""
    tr = []
    te = []
    ti = []
    for a in ran:
        print("Running the a=%f test."%(a))
        start = datetime.now()
        w = k_fold_multi(x_train, y_train, a, k, part,th)
        train_error = error_multi(x_train,softmax_y(y_train),w)
        test_error = error_multi(x_test,softmax_y(y_test),w)
        #w,v = backpropagation(x_train, x_train, y_train, y_train, a, k, hid, th)
        #train_error = nn_error(x_train,softmax_y(y_train),v,w)
        #test_error = nn_error(x_test,softmax_y(y_test),v,w)
        run_time = datetime.now() - start
        tr.append(train_error)
        te.append(test_error)
        ti.append(run_time.seconds)
        
    plt.plot(ran,tr,label='train error')
    plt.plot(ran,te,label='test error')
    plt.xlabel('test range of a')
    plt.ylabel('error rate')
    plt.title('train and test error')
    plt.show()
    
    plt.plot(ran,ti)
    plt.xlabel('test range of a')
    plt.ylabel('seconds')
    plt.title('run time')
    plt.show()

    #%%
def para_kl(x_train,x_test,y_train,y_test,a,part,ran,hid,th):
    """"hid参数nn才有"""
    tr = []
    te = []
    ti = []
    for k in ran:
        print("Running the k=%f test."%(k))
        start = datetime.now()
        w = train_valid_softmax_nok(x_train, x_train, y_train, y_train, a, k, th)
        train_error = error_multi(x_train,softmax_y(y_train),w)
        test_error = error_multi(x_test,softmax_y(y_test),w)
        #w,v = k_fold_nn(x_train, y_train, a, k,hid, part,th)
        #w,v = backpropagation(x_train, x_train, y_train, y_train, a, k, hid, th)
        #train_error = nn_error(x_train,softmax_y(y_train),v,w)
        #test_error = nn_error(x_test,softmax_y(y_test),v,w)
        run_time = datetime.now() - start
        tr.append(train_error)
        te.append(test_error)
        ti.append(run_time.seconds)
        
    plt.plot(ran,tr,label='train error')
    plt.plot(ran,te,label='test error')
    plt.xlabel('test range of k')
    plt.ylabel('error rate')
    plt.title('train and test error')
    plt.show()
    
    plt.plot(ran,ti)
    plt.xlabel('test range of k')
    plt.ylabel('seconds')
    plt.title('run time')
    plt.show()
    #%%
def para_hid(x_train,x_test,y_train,y_test,a,part,k,ran,th):
    """"hid参数nn才有"""
    tr = []
    te = []
    ti = []
    for hid in ran:
        print("Running the hid=%f test."%(hid))
        start = datetime.now()
        #w = k_fold_multi(x_train, y_train, a, k, part,th)
        #train_error = error_multi(x_train,softmax_y(y_train),w)
        #test_error = error_multi(x_test,softmax_y(y_test),w)
        #w,v = k_fold_nn(x_train, y_train, a, k,hid, part,th)
        w,v = backpropagation(x_train, x_train, y_train, y_train, a, k, hid, th)
        train_error = nn_error(x_train,softmax_y(y_train),v,w)
        test_error = nn_error(x_test,softmax_y(y_test),v,w)
        run_time = datetime.now() - start
        tr.append(train_error)
        te.append(test_error)
        ti.append(run_time.seconds)
        
    plt.plot(ran,tr,label='train error')
    plt.plot(ran,te,label='test error')
    plt.xlabel('test range of hid')
    plt.ylabel('error rate')
    plt.title('train and test error')
    plt.show()
    
    plt.plot(ran,ti)
    plt.xlabel('test range of hid')
    plt.ylabel('seconds')
    plt.title('run time')
    plt.show()
    #%%
def para_th(x_train,x_test,y_train,y_test,a,part,k,hid,ran):
    """"hid参数nn才有"""
    tr = []
    te = []
    ti = []
    for th in ran:
        print("Running the th=%f test."%(th))
        start = datetime.now()
        w = k_fold_multi(x_train, y_train, a, k, part,th)
        train_error = error_multi(x_train,softmax_y(y_train),w)
        test_error = error_multi(x_test,softmax_y(y_test),w)
        #w,v = k_fold_nn(x_train, y_train, a, k,hid, part,th)
        #train_error = nn_error(x_train,softmax_y(y_train),v,w)
        #test_error = nn_error(x_test,softmax_y(y_test),v,w)
        run_time = datetime.now() - start
        tr.append(train_error)
        te.append(test_error)
        ti.append(run_time.seconds)
        
    plt.plot(ran,tr,label='train error')
    plt.plot(ran,te,label='test error')
    plt.xlabel('test range of th')
    plt.ylabel('error rate')
    plt.title('train and test error')
    plt.show()
    
    plt.plot(ran,ti)
    plt.xlabel('test range of th')
    plt.ylabel('seconds')
    plt.title('run time')
    plt.show()
        #%%
def para_part(x_train,x_test,y_train,y_test,a,ran,k,hid,th):
    """"hid参数nn才有"""
    tr = []
    te = []
    ti = []
    for part in ran:
        print("Running the part=%f test."%(part))
        start = datetime.now()
        w = k_fold_multi(x_train, y_train, a, k, part,th)
        train_error = error_multi(x_train,softmax_y(y_train),w)
        test_error = error_multi(x_test,softmax_y(y_test),w)
        #w,v = k_fold_nn(x_train, y_train, a, k,hid, part,th)
        #train_error = nn_error(x_train,softmax_y(y_train),v,w)
        #test_error = nn_error(x_test,softmax_y(y_test),v,w)
        run_time = datetime.now() - start
        tr.append(train_error)
        te.append(test_error)
        ti.append(run_time.seconds)
        
    plt.plot(ran,tr,label='train error')
    plt.plot(ran,te,label='test error')
    plt.xlabel('test range of part')
    plt.ylabel('error rate')
    plt.title('train and test error')
    plt.show()
    
    plt.plot(ran,ti)
    plt.xlabel('test range of part')
    plt.ylabel('seconds')
    plt.title('run time')
    plt.show()
        #%%
def para_thl(x_train,x_test,y_train,y_test,a,part,k,hid,ran):
    """"hid参数nn才有"""
    tr = []
    te = []
    ti = []
    for th in ran:
        print("Running the th=%f test."%(th))
        start = datetime.now()
        w = train_valid_softmax_nok(x_train, x_train, y_train, y_train, a, k, th)
        train_error = error_multi(x_train,softmax_y(y_train),w)
        test_error = error_multi(x_test,softmax_y(y_test),w)
        #w,v = k_fold_nn(x_train, y_train, a, k,hid, part,th)
        #w,v = backpropagation(x_train, x_train, y_train, y_train, a, k, hid, th)
        #train_error = nn_error(x_train,softmax_y(y_train),v,w)
        #test_error = nn_error(x_test,softmax_y(y_test),v,w)
        run_time = datetime.now() - start
        tr.append(train_error)
        te.append(test_error)
        ti.append(run_time.seconds)
        
    plt.plot(ran,tr,label='train error')
    plt.plot(ran,te,label='test error')
    plt.xlabel('test range of th')
    plt.ylabel('error rate')
    plt.title('train and test error')
    plt.show()
    
    plt.plot(ran,ti)
    plt.xlabel('test range of th')
    plt.ylabel('seconds')
    plt.title('run time')
    plt.show()
            #%%
def para_al(x_train,x_test,y_train,y_test,ran,part,k,hid,th):
    """"hid参数nn才有"""
    tr = []
    te = []
    ti = []
    for a in ran:
        print("Running the a=%f test."%(a))
        start = datetime.now()
        w = train_valid_softmax_nok(x_train, x_train, y_train, y_train, a, k, th)
        train_error = error_multi(x_train,softmax_y(y_train),w)
        test_error = error_multi(x_test,softmax_y(y_test),w)
        #w,v = k_fold_nn(x_train, y_train, a, k,hid, part,th)
        #w,v = backpropagation(x_train, x_train, y_train, y_train, a, k, hid, th)
        #train_error = nn_error(x_train,softmax_y(y_train),v,w)
        #test_error = nn_error(x_test,softmax_y(y_test),v,w)
        run_time = datetime.now() - start
        tr.append(train_error)
        te.append(test_error)
        ti.append(run_time.seconds)
        
    plt.plot(ran,tr,label='train error')
    plt.plot(ran,te,label='test error')
    plt.xlabel('test range of a')
    plt.ylabel('error rate')
    plt.title('train and test error')
    plt.show()
    
    plt.plot(ran,ti)
    plt.xlabel('test range of a')
    plt.ylabel('seconds')
    plt.title('run time')
    plt.show()
#%%
#ran = np.array([0.05, 0.02, 0.01, 0.005])
#ran = np.array([10,5,1,0.5,0.1])
#ran = np.array([0.1])
#ran = np.arange(0.1,1.1,0.2)
#para_th(x_train_mnist,x_test_mnist,y_train_mnist,y_test_mnist,1,6,10000,1,ran)
#para_part(x_train_mnist,x_test_mnist,y_train_mnist,y_test_mnist,1,ran,10000,1,0.01)
#para_k(x_train_mnist,x_test_mnist,y_train_mnist,y_test_mnist,1,6,ran,1,0.001)
#para_a(x_train_mnist,x_test_mnist,y_train_mnist,y_test_mnist,ran,6,10000,50,0.001)
#para_k(x_train_sst,x_test_sst,y_train_sst,y_test_sst,0.01,1,ran,1,0.0001)
#para_hid(x_train_mnist,x_test_mnist,y_train_mnist,y_test_mnist,0.02,6,10000,ran,0.001)
#ran = np.array([1,0.5,0.1])
#para_al(x_train_sst,x_test_sst,y_train_sst,y_test_sst,ran,6,10000,1,0.001)
#ran = np.array([90,120,150])
#para_hid(x_train_sst,x_test_sst,y_train_sst,y_test_sst,0.002,6,10000,ran,0.001)
#ran_kl = np.array([1000,3000,5000,10000])
#para_kl(x_train_mnist,x_test_mnist,y_train_mnist,y_test_mnist,1,6,ran_kl,1,0.1)
#ran = np.array([10,50,100,150])
#para_ada(x_train_mnist,x_test_mnist,y_train_mnist,y_test_mnist,ran,1,500,50,0.01)
#ran = np.array([0.04,0.06,0.08,0.1])
#para_thl(x_train_mnist,x_test_mnist,y_train_mnist,y_test_mnist,1,6,10000,1,ran)
#%%
#a = adaboost(x_train_mnist, x_test_mnist, y_train_mnist, y_test_mnist, 10, 1, 500, 1, 6, 0.01)
#%%
def compare(x_tr,x_te,y_tr,y_te):
    tel = []
    ten = []
    tea = []
    tl = []
    tn = []
    ta = []
    for i in range(10):
        st = datetime.now()
        w = train_valid_softmax_nok(x_tr, x_tr, y_tr, y_tr, 1, 10000, 0.001)
        test_error = error_multi(x_te,softmax_y(y_te),w)
        tl.append((datetime.now()-st).seconds)        
        tel.append(test_error)
        
        st = datetime.now()
        w,v = backpropagation(x_tr, x_tr, y_tr, y_tr, 0.02, 10000, 50, 0.001)
        test_error = nn_error(x_te,softmax_y(y_te),v,w)       
        tn.append((datetime.now()-st).seconds)
        ten.append(test_error)
        
        start = datetime.now()
        ea = adaboost(x_tr, x_te, y_tr, y_te, 50, 1, 500, 50, 0.01)
        run_time = datetime.now() - start
        ta.append(run_time.seconds)
        tea.append(ea)  
    
    plt.plot(np.arange(1,11),tel,label = 'linear model')
    plt.plot(np.arange(1,11),ten,label = 'neural network')
    plt.plot(np.arange(1,11),tea,label = 'Adaboost')
    plt.xlabel('Number of runs')
    plt.ylabel('Error rate')
    plt.title('Compare the error rates of the three models')
    plt.show()
    
    plt.plot(np.arange(1,11),tl,label = 'linear model')
    plt.plot(np.arange(1,11),tn,label = 'neural network')
    plt.plot(np.arange(1,11),ta,label = 'Adaboost')
    plt.xlabel('Number of runs')
    plt.ylabel('Run time')
    plt.title('Compare the time consumption of training the three models')
    plt.show()
    
compare(x_train_mnist, x_test_mnist, y_train_mnist, y_test_mnist)        
     #%%   
def amount_data(x_tr,x_te,y_tr,y_te):
    x_tre = [0,0,0,0]
    x_tre[0] = x_tr[:5000]
    x_tre[1] = x_tr[:10000]
    x_tre[2] = x_tr[:30000]
    x_tre[3] = x_tr[:60000]
    
    y_tre = [0,0,0,0]
    y_tre[0] = y_tr[:5000]
    y_tre[1] = y_tr[:10000]
    y_tre[2] = y_tr[:30000]
    y_tre[3] = y_tr[:60000]
    
    tel = []
    ten = []
    tea = []
    tl = []
    tn = []
    ta = []
    for i in range(4):
        
        st = datetime.now()
        w = train_valid_softmax_nok(x_tre[i], x_tre[i], y_tre[i], y_tre[i], 1, 10000, 0.001)
        test_error = error_multi(x_te,softmax_y(y_te),w)
        tl.append((datetime.now()-st).seconds)        
        tel.append(test_error)
        
        st = datetime.now()
        w,v = backpropagation(x_tre[i], x_tre[i], y_tre[i], y_tre[i], 0.02, 10000, 50, 0.001)
        test_error = nn_error(x_te,softmax_y(y_te),v,w)       
        tn.append((datetime.now()-st).seconds)
        ten.append(test_error)
        
        start = datetime.now()
        ea = adaboost(x_tre[i], x_te, y_tre[i], y_te, 50, 1, 500, 50, 0.01)
        run_time = datetime.now() - start
        ta.append(run_time.seconds)
        tea.append(ea)  
    
    xaxis = np.array([5000,10000,30000,60000])
    plt.plot(xaxis,tel,label = 'linear model')
    plt.plot(xaxis,ten,label = 'neural network')
    plt.plot(xaxis,tea,label = 'Adaboost')
    plt.xlabel('Size of training set')
    plt.ylabel('Error rate')
    plt.title('Influence of training data size on error rate')
    plt.show()
    
    plt.plot(xaxis,tl,label = 'linear model')
    plt.plot(xaxis,tn,label = 'neural network')
    plt.plot(xaxis,ta,label = 'Adaboost')
    plt.xlabel('Size of training set')
    plt.ylabel('Run time')
    plt.title('Influence of training data size on training time')
    plt.show()
    
amount_data(x_train_mnist, x_test_mnist, y_train_mnist, y_test_mnist)
    
        
        
        
#%%
def train_valid_softmax(x_train,x_validation,y_train,y_validation,a,k,threshold):
    """Linear model for multiple-class classification."""
    #print("y_train:",y_train.shape)
    dx = x_train.shape[1]
    dy = y_train.shape[1]
    w = np.random.random((x_train.shape[1],y_train.shape[1]))
    #print(error_multi(x_validation, y_validation, w))
    l_new = 0
    l_old = 1
    c = 1
#    while l_new <= l_old:
    while abs(l_new - l_old) >= threshold:
        l_old = l_new          
        for i in range(k):
            t = random.randint(0, len(x_train)-1)
            p = fwx(x_train[t],w)
            #print("alpha:",np.dot(x_train[t],w),"p:",p)
            #break
            w += a*np.dot(x_train[t].reshape(dx,1),(y_train[t]-p).reshape(1,dy))
            #w = w + a*((y_train[t]-fwx(w.T,x_train[t]))*(x_train[t].reshape(1,len(x_train[t])).T))
        print("Have updated w %d times."%(c))
        l_new = error_multi(x_validation,y_validation,w)        
        #print(l_new)
        c += 1        
    #print("www:",error_multi(x_validation,y_validation,w))
    print("Finished learning. The min error on this validation set is %f"%(l_old))
    return w
   #%%
def train_valid_softmax_nok_dx(x_train,x_validation,y_train,y_validation,a,k,weightx,threshold):
    """Linear model for multiple-class classification."""
    #print("y_train:",y_train.shape)
    y_train = softmax_y(y_train)
    y_validation = softmax_y(y_validation)
    dx = x_train.shape[1]
    dy = y_train.shape[1]
    w = np.random.random((x_train.shape[1],y_train.shape[1]))
    #print(error_multi(x_validation, y_validation, w))
    l_new = 0
    l_old = 1
    c = 1
#    while l_new <= l_old:
    while abs(l_new - l_old) >= threshold:
        l_old = l_new          
        for i in range(k):
            t = random.randint(0, len(x_train)-1)
            p = fwx(x_train[t],w)
            #print("alpha:",np.dot(x_train[t],w),"p:",p)
            #break
            w += a*np.dot(x_train[t].reshape(dx,1),(y_train[t]-p).reshape(1,dy))
            #w = w + a*((y_train[t]-fwx(w.T,x_train[t]))*(x_train[t].reshape(1,len(x_train[t])).T))
        print("Have updated w %d times."%(c))
        l_new = error_multi_p(x_validation,y_validation,w,weightx)[0]        
        #print(l_new)
        c += 1        
    #print("www:",error_multi(x_validation,y_validation,w))
    print("Finished learning. The min error on this validation set is %f"%(l_old))
    return w
    #%%
def train_valid_softmax_nok(x_train,x_validation,y_train,y_validation,a,k,threshold):
    """Linear model for multiple-class classification."""
    #print("y_train:",y_train.shape)
    y_train = softmax_y(y_train)
    y_validation = softmax_y(y_validation)
    dx = x_train.shape[1]
    dy = y_train.shape[1]
    w = np.random.random((x_train.shape[1],y_train.shape[1]))
    #print(error_multi(x_validation, y_validation, w))
    l_new = 0
    l_old = 1
    c = 1
#    while l_new <= l_old:
    while abs(l_new - l_old) >= threshold:
        l_old = l_new          
        for i in range(k):
            t = random.randint(0, len(x_train)-1)
            p = fwx(x_train[t],w)
            #print("alpha:",np.dot(x_train[t],w),"p:",p)
            #break
            w += a*np.dot(x_train[t].reshape(dx,1),(y_train[t]-p).reshape(1,dy))
            #w = w + a*((y_train[t]-fwx(w.T,x_train[t]))*(x_train[t].reshape(1,len(x_train[t])).T))
        print("Have updated w %d times."%(c))
        l_new = error_multi(x_validation,y_validation,w)        
        #print(l_new)
        c += 1        
    #print("www:",error_multi(x_validation,y_validation,w))
    print("Finished learning. The min error on this validation set is %f"%(l_old))
    return w
#%%
def train_valid_softmax_dx(x_train,x_validation,y_train,y_validation,a,k,weightx,threshold):
    """Linear model for multiple-class classification with different sample weights."""
    #print(y_train.shape)
    w = np.random.random((x_train.shape[1],y_train.shape[1]))
    dx = x_train.shape[1]
    dy = y_train.shape[1]
    #print(error_multi(x_validation, y_validation, w))
    l_new = 0
    l_old = 1
    c = 1
    while abs(l_new - l_old) >= threshold:
        l_old = l_new          
        for i in range(k):
            t = random.randint(0, len(x_train)-1)            
            #p = fwx(x_train[t],w)
            p = fwx(x_train[t],w)
            w += weightx[t]*a*np.dot(x_train[t].reshape(dx,1),(y_train[t]-p).reshape(1,dy))
            #w += weightx[t]*a*np.dot(x_train[t].reshape(dx,1),(y_train[t]-p).reshape(1,dy))
        print("Have updated w %d times."%(c))
        l_new = error_multi_p(x_validation,y_validation,w,weightx)[0]        
        #print(l_new)
        c += 1        
    print("Finished learning. The min error on this validation set is %f"%(l_old))
    return w  
#%%
"""Adaboost"""
def adaboost(x_tr, x_te, y_tr, y_tee, t, a, k, hid, th):
    y_te = softmax_y(y_tee)
    #dx = 1/len(x_tr)*np.ones((len(x_tr),1))
    dx = np.ones((len(x_tr),1))
    alpha = np.zeros((t,1))
    #b = np.zeros(t,1)
    hh = np.zeros((len(x_te),y_te.shape[1]))
    #y = y_ada(y)
    for i in range(t):
        w = train_valid_softmax_nok_dx(x_tr,x_tr, y_tr,y_tr, a, k,dx,th)
        er,yorn= error_multi_pp(x_tr, softmax_y(y_tr), w,dx)
        #print("eeeeeeeeeeeeeeeee",er)
        if er <= 0.5:
            #print("Break the %dth classifier."%(i))           
            alpha[i] = 0.5*np.log((1-er)/er)
            dx = dx_update(dx,alpha[i],yorn)
            hh += p_linear(x_te,y_te,w)*alpha[i]
            print("The %dth simple classifier has error rate %f"%(i,er))
        
    hh = ada_predict(hh)
    ea = sperror(hh,y_tee)
    print("Adaboost ---done. The error is %f"%(ea))
    return ea
        #%%
def error_multi_pp(x,y,w,dx):
    l = 0
    p = predict_multilabels(x, w)
    y = lower_dim_y(y)
   #print("p,y",p.shape,y.shape)
    yorn = np.ones(len(y))
    #print(y[:10],p[:10])
    for i in range(len(y)):              
        if y[i] != p[i]:
            l += 1
            yorn[i] = -1
    r = l/len(y)
    print("The error rate is %f"%(r))
    return r,yorn
#%%
"""Adaboost"""
def ada_plot(x_tr, x_te, y_tr, y_tee, t, a, k, hid, th):
    start = datetime.now()
    y_te = softmax_y(y_tee)
    #dx = 1/len(x_tr)*np.ones((len(x_tr),1))
    dx = np.ones((len(x_tr),1))
    alpha = np.zeros((t,1))
    #b = np.zeros(t,1)
    hh = np.zeros((len(x_te),y_te.shape[1]))
    #y = y_ada(y)
    el = []
 
    for i in range(t):
        w = train_valid_softmax_nok_dx(x_tr,x_tr, y_tr,y_tr, a, k,dx,th)
        er,yorn= error_multi_pp(x_tr, softmax_y(y_tr), w,dx)
        el.append(er)
        #print("eeeeeeeeeeeeeeeee",er)
        if er <= 0.5:
            #print("Break the %dth classifier."%(i))           
            alpha[i] = 0.5*np.log((1-er)/er)
            dx = dx_update(dx,alpha[i],yorn)
            hh += p_linear(x_te,y_te,w)*alpha[i]
            print("The %dth simple classifier has error rate %f"%(i,er))

    hh = ada_predict(hh)
    ea = sperror(hh,y_tee)
    print("Adaboost run time:%d"%((datetime.now()-start).seconds))
    plt.plot(np.arange(1,t+1),el,label = 'Weak classifiers')
    plt.plot(np.arange(1,t+1),ea*np.ones(t),label = 'Adaboost')
    plt.xlabel('Number of weak classifiers')
    plt.ylabel('error rate')
    plt.title('Adaboost model')
    plt.show()
    
    print("Adaboost ---done. The error is %f"%(ea))
    return ea
#%%
ada_plot(x_train_mnist, x_test_mnist, y_train_mnist, y_test_mnist, 50, 1, 500, 1, 0.01)
        #%%
def para_ada(x_train,x_test,y_train,y_test,ran,a,k,hid,th):
    """"hid参数nn才有"""
    te = []
    ti = []
    for t in ran:
        print("Running the t=%f test."%(t))
        start = datetime.now()
        ea = adaboost(x_train, x_test, y_train, y_test, t, a, k, hid, th)
        run_time = datetime.now() - start
        ti.append(run_time.seconds)
        te.append(ea)       
        
    plt.plot(ran,te,label='test error')
    plt.xlabel('test range of t')
    plt.ylabel('error rate')
    plt.title('test error')
    plt.show()
    
    plt.plot(ran,ti)
    plt.xlabel('test range of t')
    plt.ylabel('seconds')
    plt.title('run time')
    plt.show()
#%%
def sperror(p,y):
    l = 0
    for i in range(len(y)):       
        if y[i] != p[i]:
            l += 1
    r = l/len(y)
    print("Function sperror ---done. The error rate is %f"%(r))
    return r
#%%
def p_linear(x,y,w):
    y = fwx(x,w)
    p = -np.ones(y.shape)
    for i in range(len(y)):
        p[i][np.argmax(y[i])] = 1
    return p

def p_nn(x,v,w):
    h = fwx(x,v)   
    y = fwx(h,w)
    p = -np.ones(y.shape)
    for i in range(len(y)):
        p[i][np.argmax(y[i])] = 1
    return p    
  
#%%
def ada_predict(hh):
    p = np.zeros(len(hh))
    for i in range(len(hh)):
        p[i] = np.argmax(hh[i])
    return p
#%%
def k_fold_nn_dx(x,y,a,k,hid,part,dx,th):
    """A k_fold function for neural networks with different sample weights."""
    #print(y.shape)
    y_max = softmax_y(y)
    partition = len(x)//part
    w = []
    v = []
    for i in range(part):
        x_validation = x[part:part+partition]
        y_validation = y_max[part:part+partition]
        x_train = np.concatenate((x[:part],x[(part+partition):]),axis = 0)
        y_train = np.concatenate((y_max[:part],y_max[(part+partition):]),axis = 0)
        #print(y_train.shape,y_max)
        w_k,v_k = backpropagation_dx(x_train,x_validation,y_train,y_validation,a,k,hid,dx,th)
        
        w.append(w_k)
        v.append(v_k)
        print("k_fold_nn: Finished the %dth fold_multi validation."%(i))
    w_all = np.zeros(w[0].shape)
    v_all = np.zeros(v[0].shape)
    for i in range(len(w)):
        w_all += w[i]
        v_all += v[i]
    w_ave = w_all/part
    v_ave = v_all/part
    print("Function k_fold_nn_dx: done. The error rate on the whole training set is %f."%(nn_error(x, y_max, v_ave, w_ave)))
    return w_ave,v_ave
#%%
def backpropagation_dx(x_train,x_validation,y_train,y_validation,a,k,hid,weightx,threshold):
    """"Grediant descent method for feedforward neural networks with one hidden layer
    with different sample weigths."""
    v = np.random.random((x_train.shape[1],hid))
    w = np.random.random((hid,y_train.shape[1]))
    l_new = 1
    l_old = 0
    c = 1
    yd = len(y_train[0])
    dx = len(x_train[0])
    while abs(l_new - l_old) >= threshold:
        l_old = l_new                    
        for i in range(k):         
            t = random.randint(0, len(x_train)-1)
            h = fwx(x_train,v)
            pre = fwx(h,w)
            #print(pre.shape)
            w += weightx[t]*np.dot(h[t].reshape(hid,1),((y_train[t]-pre[t])*np.dot((pre[t].reshape(1,yd)),(1-pre[t]).reshape(yd,1))))
            #print((y_train[t]-pre[t]).T.shape)
            eph = -np.dot(w,((y_train[t]-pre[t]).reshape(yd,1))*(np.dot(pre[t].reshape(1,yd),(1-pre[t]).reshape(yd,1))))
            v -= weightx[t]*np.dot(x_train[t].reshape(dx,1),(eph.T)*(np.dot(h[t].reshape(1,hid),(1-h[t]).reshape(hid,1))))
            #print(y_train[0].shape,x_train[0].T.shape,w.shape,fwx(w.T,x_train[t]).shape)
            print("Tring %d in %d times."%(i,k))
            #w += a*(d[i]-y_train[i])*(y_train[i].reshape(1,len(y_train[i])))*(1-y_train[i])*h
        print("Backpropgation: Have updated w %d times."%(c))
        l_new = nn_error_p(x_validation,y_validation,v,w,dx)        
        #print(l_new)
        c += 1        
    print("Function backpropagation ---done. The min error on this validation set is %f"%(l_old))
    return w,v
#%%
def nn_error_p(x,y,v,w,dx):
    pre = nn_predict(x, v, w)
    yone = lower_dim_y(y)
    error = 0
    yorn = np.ones(len(y))
    #print(pre[:10],yone[:10])
    for i in range(len(pre)):
        if pre[i] != yone[i]:
            error += dx[i]
            yorn[i] = -1
    #print(error,len(pre))
    er_rate = error/len(pre)
    print("Function nn_error ---done. The error rate is %f"%(er_rate))
    return er_rate,pre,yorn
        #%%
def error_multi_p(x,y,w,dx):
    l = 0
    p = predict_multilabels(x, w)
    y = lower_dim_y(y)
   #print("p,y",p.shape,y.shape)
    yorn = np.ones(len(y))
    #print(y.shape)
    for i in range(len(y)):              
        if y[i] != p[i]:
            l += dx[i]
            yorn[i] = -1
    r = l/len(y)
    print("The error rate is %f"%(r))
    return r,p,yorn
#%%
def y_ada(y):
    """"类似于softmax_y()"""
    max = np.int(y.max())
    y_max = -np.ones((len(y),max+1))
    for i in range(len(y_max)):
        y_max[i][np.int(y[i])]  = 1
    print("Have changed y into an array with shape",y_max.shape)
    return y_max   
#%%
def k_fold_multi_dx(x,y,a,k,part,dx,th):
    """"k-fold function for multiple-class linear classification 
    with different sample weights."""
    #print(y.shape)
    y_max = softmax_y(y)
    partition = len(x)//part
    w = []
    for i in range(part):
        x_validation = x[part:part+partition]
        y_validation = y_max[part:part+partition]
        x_train = np.concatenate((x[:part],x[(part+partition):]),axis = 0)
        y_train = np.concatenate((y_max[:part],y_max[(part+partition):]),axis = 0)
        print("x_train",x_train.shape)
        w_k = train_valid_softmax_dx(x_train,x_validation,y_train,y_validation,a,k,dx,th)
        w.append(w_k)
        print("Finished the %dth fold_multi_dx validation."%(i))
    w_all = np.zeros(w[0].shape)
    for i in range(len(w)):
        w_all += w[i]
    w_ave = w_all/part
    er_ave = error_multi(x, y_max, w_ave)
    print("Function k_fold_multi: done. Error on the whole training set is %f"%(er_ave))
    return w_ave    
 
#%%
def dx_update(dx,a,yorn):
    for i in range(len(dx)):
        dx[i] = dx[i]*np.exp(-a*yorn[i])
    z = sum(dx)
    for i in range(len(dx)):
        dx[i] = (dx[i]/z)*len(dx)
    return dx

    
    
