# -*- coding:utf-8 -*-  

import sys, os
import numpy as np
import cv2

#把稠密数据label[1,5...]变为[[0,1,0,0...],[...]...]
def dense_to_one_hot(labels_dense, num_classes):
    #数据数量
    num_labels = labels_dense.shape[0]
    #生成[0,1,2...]*10,[0,10,20...]
    index_offset = np.arange(num_labels) * num_classes
    #初始化np的二维数组
    labels_one_hot = np.zeros((num_labels, num_classes))
    #相对应位置赋值变为[[0,1,0,0...],[...]...]
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

def isPathExist(filePath):
    if os.path.exists(filePath) :
        if not filePath.endswith(".npy"):
            raise ValueError('Please input file path end with ".npy"')
    else :
        raise ValueError('error, can not find file: '+filePath)
    return True

class GetMaleFemaleData(object):
    """docstring for GetMaleFemaleData"""
    def __init__(self):
        super(GetMaleFemaleData, self).__init__()
        dog_train_data_path = 'training_set/dogs/all_image.npy'
        cat_train_data_path = 'training_set/cats/all_image.npy'
        dog_test_data_path = 'test_set/dogs/all_image.npy'
        cat_test_data_path = 'test_set/cats/all_image.npy'
        bg_path = 'background_image32X32.npy'
        isPathExist(dog_train_data_path)
        isPathExist(cat_train_data_path)
        isPathExist(dog_test_data_path)
        isPathExist(cat_test_data_path)
        isPathExist(bg_path)
        dog_train_data = np.load(dog_train_data_path)
        cat_train_data = np.load(cat_train_data_path)
        dog_test_data = np.load(dog_test_data_path)
        cat_test_data = np.load(cat_test_data_path)
        bg_data = np.load(bg_path)
        print dog_train_data.shape, cat_train_data.shape, dog_test_data.shape, cat_test_data.shape, bg_data.shape

        self.train_data = np.vstack((dog_train_data,cat_train_data,bg_data[0:-10001]))
        self.train_lable = np.vstack((dense_to_one_hot(np.ones(dog_train_data.shape[0]+cat_train_data.shape[0], dtype = np.uint8),2), 
            dense_to_one_hot(np.zeros(bg_data.shape[0]-10000, dtype = np.uint8),2)))
        self.test_data = np.vstack((dog_test_data,cat_test_data,bg_data[-10001:-1]))
        self.test_lable =  np.vstack((dense_to_one_hot(np.ones(dog_test_data.shape[0]+cat_test_data.shape[0], dtype = np.uint8),2),
            dense_to_one_hot(np.zeros(10000, dtype = np.uint8),2)))
        print 'self.train_data.shape = '+str(self.train_data.shape)
        print 'self.train_lable.shape = '+str(self.train_lable.shape)
        print 'self.test_data.shape = '+str(self.test_data.shape)
        print 'self.test_lable.shape = '+str(self.test_lable.shape)
        self.train_data_index = 0
        self.test_data_index = 0

    def next_train_batch(self, batch_size):
        if self.train_data_index+batch_size >= self.train_data.shape[0]:
            self.train_data_index = 0
        if self.train_data_index == 0:
            perm = np.arange(self.train_data.shape[0])
            np.random.shuffle(perm)
            self.train_data = self.train_data[perm]
            self.train_lable = self.train_lable[perm]
        self.train_data_index = self.train_data_index+batch_size
        return self.train_data[self.train_data_index-batch_size:self.train_data_index], self.train_lable[self.train_data_index-batch_size:self.train_data_index]

    def get_test_data(self):
        return self.test_data, self.test_lable
        pass

    def next_test_batch(self, batch_size):
        if self.test_data_index+batch_size >= self.test_data.shape[0]:
            self.test_data_index = 0
        if self.test_data_index == 0:
            perm = np.arange(self.test_data.shape[0])
            np.random.shuffle(perm)
            self.test_data = self.test_data[perm]
            self.test_lable = self.test_lable[perm]
        self.test_data_index = self.test_data_index+batch_size
        return self.test_data[self.test_data_index-batch_size:self.test_data_index], self.test_lable[self.test_data_index-batch_size:self.test_data_index]
        pass

def main():
    # male_dir = '../../data/FemaleMaleFace_30x30/1_Male.npy' #sys.argv[1]#'/home/ysq/YSQWork/DeepLearning/data/FemaleMaleFace_30x30/1_Male.npy'
    # female_dir = '../../data/FemaleMaleFace_30x30/0_Female.npy' #sys.argv[2]#'/home/ysq/YSQWork/DeepLearning/data/FemaleMaleFace_30x30/0_Female.npy'
    data_get = GetMaleFemaleData()
    train_data, train_lable = data_get.next_train_batch(50)
    # train_data, train_lable = data_get.get_test_data()
    cv2.namedWindow('image',cv2.WINDOW_FULLSCREEN)
    for i in xrange(train_data.shape[0]):
        cv2.imshow('image', train_data[i])
        print train_lable[i]
        cv2.waitKey(0)
    pass

if __name__ == '__main__':
    main()
