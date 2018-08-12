#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import getopt
import numpy as np
import cv2
import json

input_dir = ''
output_dir = ''


class ImageLable(object):
    """docstring for ImageLable"""

    def __init__(self, input_path):
        super(ImageLable, self).__init__()
        self.input_path_ = input_path
        # self.output_path_ = output_path
        self.file_list_ = None
        self.lable_start_x_ = 0
        self.lable_start_y_ = 0
        self.lable_stop_x_ = 0
        self.lable_stop_y_ = 0
        self.image_ = None
        self.drawing_ = False
        self.lable_data = []
        self.lable_file_name = input_path.split('/')[-2]+"_lable.json"
        self.image_lable_num_ = 0
        if not os.path.exists(self.input_path_):
            raise ValueError('Can not find input path: "'+self.input_path_+'"')

    def begin(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.image_lable)
        self.load_image()
        for i in range(0, len(self.file_list_)):
            path = os.path.join(self.input_path_, self.file_list_[i])
            if os.path.isfile(path) and (path.endswith('.jpg') or path.endswith('.png')):
                # 你想对文件的操作
                self.image_ = cv2.imread(path)
                cv2.imshow('image', self.image_)
                while True:
                    image_cp = self.image_.copy()
                    # cv2.imwrite(os.path.join(output_dir, self.file_list_[i]), self.image_)
                    key_value = cv2.waitKey(0)
                    if key_value & 0xFF == 27:  # ESC
                        break
                        # sys.exit(0)
                    elif key_value == 119:   # w
                        if self.lable_start_y_ > 0 and self.lable_start_x_ > 0:
                            self.lable_start_y_ = self.lable_start_y_-1
                            self.lable_start_x_ = self.lable_start_x_-1
                    elif key_value == 115:  # s
                        if self.lable_start_y_ < self.lable_stop_y_-10 and self.lable_start_x_ < self.lable_stop_x_-10:
                            self.lable_start_y_ = self.lable_start_y_+1
                            self.lable_start_x_ = self.lable_start_x_+1
                    elif key_value == 97:   # a
                        if self.lable_stop_y_-10 > self.lable_start_y_ and self.lable_stop_x_ - 10 > self.lable_start_x_:
                            self.lable_stop_y_ = self.lable_stop_y_-1
                            self.lable_stop_x_ = self.lable_stop_x_-1
                    elif key_value == 100:  # d
                        if self.lable_stop_y_ < image_cp.shape[0] and self.lable_stop_x_ < image_cp.shape[1]:
                            self.lable_stop_y_ = self.lable_stop_y_+1
                            self.lable_stop_x_ = self.lable_stop_x_+1
                    elif  key_value == 87:  # W
                        if self.lable_start_y_ > 0:
                            self.lable_start_y_ = self.lable_start_y_-1
                            self.lable_stop_y_ = self.lable_stop_y_-1
                    elif  key_value == 83:  # S
                        if self.lable_stop_y_ < image_cp.shape[0]:
                            self.lable_start_y_ = self.lable_start_y_+1
                            self.lable_stop_y_ = self.lable_stop_y_+1
                    elif  key_value == 65:  # A
                        if self.lable_start_x_ > 0:
                            self.lable_start_x_ = self.lable_start_x_-1
                            self.lable_stop_x_ = self.lable_stop_x_-1
                    elif  key_value == 68:  # D
                        if self.lable_stop_x_ < image_cp.shape[1]:
                            self.lable_start_x_ = self.lable_start_x_+1
                            self.lable_stop_x_ = self.lable_stop_x_+1
                    elif key_value == 32:  # Blank
                        # Blabk, save value
                        self.image_lable_num_ = self.image_lable_num_+1
                        print 'save image lable: '+self.file_list_[i]
                        self.lable_data.append({'image_name': self.file_list_[i],
                            'start_x':self.lable_start_x_,
                            'start_y':self.lable_start_y_,
                            'stop_x':self.lable_stop_x_,
                            'stop_y':self.lable_stop_y_})
                        file_path = os.path.join(self.input_path_, self.lable_file_name)
                        with open(file_path,"w") as f:
                            json.dump({'lable_data':self.lable_data},f)
                        print '    lable saved in: '+ file_path+', image number: '+str(self.image_lable_num_)
                        break
                    else:
                        print key_value
                        continue
                    cv2.rectangle(image_cp, (self.lable_start_x_, self.lable_start_y_),
                                  (self.lable_stop_x_, self.lable_stop_y_), (0, 0, 255), 3)
                    cv2.imshow('image', image_cp)
            else :
                print 'jump image:'+path
        pass

    def load_image(self):
        self.file_list_ = os.listdir(self.input_path_)  # 列出文件夹下所有的目录与文件
        pass

    def image_lable(self, event, x, y, flags, param):
        if x > self.image_.shape[1]:
            x = self.image_.shape[1]
        if y > self.image_.shape[0]:
            y = self.image_.shape[0]
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing_ = True
            self.lable_start_x_ = x
            self.lable_start_y_ = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing_ = False
            width = x - self.lable_start_x_
            height = y - self.lable_start_y_
            if width > height:
                width = height
            else:
                height = width
            self.lable_stop_x_ = self.lable_start_x_+width
            self.lable_stop_y_ = self.lable_start_y_+height
            image_cp = self.image_.copy()
            cv2.rectangle(image_cp, (self.lable_start_x_, self.lable_start_y_),
                          (self.lable_stop_x_, self.lable_stop_y_), (0, 0, 255), 3)
            # cv2.circle(self.image_, (self.lable_start_x_-10,
            #                          self.lable_start_y_-20), 3, (0, 0, 255), -1)
            cv2.imshow('image', image_cp)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing_ == True:
            width = x - self.lable_start_x_
            height = y - self.lable_start_y_
            if width > height:
                width = height
            else:
                height = width
            self.lable_stop_x_ = self.lable_start_x_+width
            self.lable_stop_y_ = self.lable_start_y_+height
            image_rec = np.zeros(self.image_.shape, np.uint8)
            cv2.rectangle(image_rec, (self.lable_start_x_, self.lable_start_y_),
                          (self.lable_stop_x_, self.lable_stop_y_), (0, 0, 255), 3)
            image_add = cv2.addWeighted(image_rec, 1, self.image_, 1, 0.0)
            cv2.imshow('image', image_add)

        pass

class LoadImageLable(object):
    """docstring for LoadImageLable"""
    def __init__(self, input_path):
        super(LoadImageLable, self).__init__()
        self.input_path_ = input_path
        self.lable_data_ = []
        self.lable_file_name_ = input_path.split('/')[-2]+"_lable.json"
        self.image_lable_num_ = 0

    def begin(self):
        load_image()
        for lable in self.lable_data_:
            image_path = os.path.join(self.input_path_, lable['image_name'])
            self.image_lable_num_ = self.image_lable_num_+1
            print image_path+"\n\timage number :"+str(self.image_lable_num_)
            start_x = lable['start_x']
            start_y = lable['start_y']
            stop_x = lable['stop_x']
            stop_y = lable['stop_y']
            image = cv2.imread(image_path)
            cv2.rectangle(image, (start_x, start_y), (stop_x, stop_y), (0, 255, 0), 3)
            cv2.imshow('lable image', image)
            cv2.waitKey(0)
            pass
        pass
    def load_image(self):
        file_path = os.path.join(self.input_path_, self.lable_file_name_)
        with open(file_path,'r') as load_f:
            load_dict = json.load(load_f)
            self.lable_data_ = load_dict['lable_data']
            return self.lable_data_


def path_init():
    global input_dir, output_dir
    opts, args = getopt.getopt(sys.argv[1:], "hi:o:a:", ["version=", "aha="])
    for op, value in opts:
        if op == "-i":
            input_dir = value
            if not os.path.exists(input_dir):
                raise ValueError("Input Path Error: "+input_dir +
                                 "/  , Can not find such folder.")
        elif op == "-o":
            output_dir = value
            # print 'output_dir = '+value
        # elif op == "-a":
        #     print 'a = '+value
        # elif op == "--version":
        #     print 'version = '+value
        # elif op == "--aha":
        #     print 'aha = '+value
        elif op == "-h":
            print "-i image input path\r\n-o image output path"
            print "Lable :"
            print "w: Top line upward 1 pixel"
            print "s: Top line down 1 pixel"
            print "a: Left line turn left 1 pixel"
            print "d: Left line turn right 1 pixel"
            print "W: Bottom line upward 1 pixel"
            print "S: Bottom line down 1 pixel"
            print "A: Right line turn left 1 pixel"
            print "D: Right line turn right 1 pixel"
            sys.exit()
    pass


        

def main():
    # path_init()
    # lable_img = ImageLable(input_dir)
    # lable_img.begin()
    
    # read_lable_imge = LoadImageLable(input_dir)
    # read_lable_imge.begin()


    file_list_ = os.listdir('../../data/0_bg')  # 列出文件夹下所有的目录与文件
    image_save = []
    for i in range(0, len(file_list_)):
        path = os.path.join('../../data/0_bg', file_list_[i])
        if os.path.isfile(path) and (path.endswith('.jpg') or path.endswith('.png')):
            # 你想对文件的操作
            image_src = cv2.imread(path)
            # cv2.imshow('image', image_src)
            # cv2.waitKey(0)
            image_save.append(cv2.resize(
                            image_src, (32, 32), interpolation=cv2.INTER_CUBIC))
            if i%1000 == 0:
                print i
    np.save('background_image32X32.npy', image_save)
    print 'Saved OK.'

    # lable_data = []
    # lable_data.append({'name':'hello', 'x':10,'y':10})
    # lable_data.append({'name':'hello2', 'x':20,'y':20})
    # print lable_data
    # j_lable_data={'lable_data':lable_data}
    # print j_lable_data
    # with open("test_lable.json","w") as f:
    #     json.dump(j_lable_data,f)
    # with open("test_lable.json",'r') as load_f:
    #     load_dict = json.load(load_f)
    #     print(load_dict)
    #     print load_dict['lable_data']


if __name__ == '__main__':
    main()
