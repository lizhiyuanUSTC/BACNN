import numpy as np
import tensorflow as tf
import os
import cv2
import struct
import pickle
import HCCR_FLAGS
import math

FLAGS = HCCR_FLAGS.FLAGS
def resize(img):
    new_img = np.zeros((FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE), dtype=np.uint8)
    height, width = img.shape
    new_height = new_width = FLAGS.IMAGE_SIZE
    if width < height:
        R1 = width * 1.0 / height
        R2 = math.sqrt(math.sin(math.pi*R1/2))
        new_width = int(FLAGS.IMAGE_SIZE * R2)
        if new_width % 2 == 1:
            new_width += 1
        start = int((FLAGS.IMAGE_SIZE - new_width) / 2)
        end = int(start + new_width)
        _img = cv2.resize(img, (new_width, new_height))
        new_img[:, start:end] = _img
    elif width == height:
        new_img = cv2.resize(img, (new_width, new_height))
    else:
        R1 = height * 1.0 / width
        R2 = math.sqrt(math.sin(math.pi*R1/2))
        new_height = int(FLAGS.IMAGE_SIZE * R2)
        if new_height % 2 == 1:
            new_height += 1
        start = int((FLAGS.IMAGE_SIZE - new_height) / 2)
        end = int(start + new_height)
        _img = cv2.resize(img, (new_width, new_height))
        new_img[start:end, :] = _img
    return new_img

def one_file(file_path, isAug=False):
    kernelX = np.ones((1,3), np.uint8)
    kernelY = np.ones((3,1), np.uint8)
    with open(file_path, "rb") as f:
        header_size = 10
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size: break
            sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)
            tagcode = header[5] + (header[4]<<8)
            width = header[6] + (header[7]<<8)
            height = header[8] + (header[9]<<8)
            if header_size + width*height != sample_size:
                break
            image = np.fromfile(f, dtype='uint8', count=width*height).reshape((height, width))
            yield image, tagcode
            if isAug:
                image_erodeY = cv2.erode(image, kernelY)
                yield image_erodeY, tagcode

                image_dilateY = cv2.dilate(image, kernelY)
                yield image_dilateY, tagcode

                image_erodeX = cv2.erode(image, kernelX)
                yield image_erodeX, tagcode

                image_dilateX = cv2.dilate(image, kernelX)
                yield image_dilateX, tagcode

                image_erodeX_erodeY = cv2.erode(image_erodeX, kernelY)
                yield image_erodeX_erodeY, tagcode

                #image_erodeY_erodeX = cv2.erode(image_erodeY, kernelX)
                #yield image_erodeY_erodeX, tagcode

                image_erodeX_dilateY = cv2.dilate(image_erodeX, kernelY)
                yield image_erodeX_dilateY, tagcode

                #image_dilateY_erodeX = cv2.erode(image_dilateY, kernelX)
                #yield image_dilateY_erodeX, tagcode

                image_dilateX_erodeY = cv2.erode(image_dilateX, kernelY)
                yield image_dilateX_erodeY, tagcode

                #image_erodeY_dilateX = cv2.dilate(image_erodeY, kernelX)
                #yield image_erodeY_dilateX, tagcode

                image_dilateX_dilateY = cv2.dilate(image_dilateX, kernelY)
                yield image_dilateX_dilateY, tagcode

                #image_dilateY_dilateX = cv2.dilate(image_dilateY, kernelX)
                #yield image_dilateY_dilateX, tagcode




def read_from_gnt_dir(gnt_dir, isAug=False):
    for file_name in sorted(os.listdir(gnt_dir)):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dir, file_name)
            print(file_path)
            for image, tagcode in one_file(file_path, isAug):
                yield image, tagcode

def saveImage(gnt_dir, writer, data_version=1.1, counter = 0, isAug=False):
    print(gnt_dir)
    if data_version == 1.1:
        read_func = read_from_gnt_dir
    elif data_version == 1.0:
        read_func = one_file
    for image, tagcode in read_func(gnt_dir, isAug):
        image = 255 - image
        image = resize(image)
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_raw = image.tobytes()
        tagcode = struct.pack('>H', tagcode).decode('gb2312')
        example = tf.train.Example(features=tf.train.Features(feature={
                            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[char_dict[tagcode]]))}))
        writer.write(example.SerializeToString())
        
        counter += 1
        if counter % 10000 == 0:
            print(" %7d files has been saved!" % counter)
    return counter
 
if not os.path.exists('char_dict'):            
    char_set = set()
    for _, tagcode in read_from_gnt_dir("/media/ai/DL_DATA/HCCR/original_data/1.1train-gb1"):
        tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
        char_set.add(tagcode_unicode)
    char_list = list(char_set)
    char_dict = dict(zip(sorted(char_list), range(len(char_list))))
    label2char = dict(zip(range(len(char_list)), sorted(char_list)))
    print(len(char_dict))
    with open('char_dict', 'wb') as f:
        pickle.dump(char_dict, f)
    with open('label2char', 'wb') as f:
        pickle.dump(label2char, f)
else:
    with open('char_dict', 'rb') as f:
        char_dict = pickle.load(f)


train_writer = tf.python_io.TFRecordWriter(FLAGS.train_data)
train_num = saveImage("/media/ai/DL_DATA/HCCR/original_data/1.1train-gb1/", train_writer, isAug=False)
train_num = saveImage("/media/ai/DL_DATA/HCCR/original_data/1.0train-gb1.gnt", train_writer, data_version=1.0, counter=train_num, isAug=False)
#train_writer.close()


#test_writer = tf.python_io.TFRecordWriter(FLAGS.test_data)
train_num = saveImage("/media/ai/DL_DATA/HCCR/original_data/1.1test-gb1/", train_writer, counter=train_num, isAug=False)
train_num = saveImage("/media/ai/DL_DATA/HCCR/original_data/1.0test-gb1.gnt", train_writer, data_version=1.0, counter=train_num, isAug=False)
train_writer.close()
'''
competition_writer = tf.python_io.TFRecordWriter(FLAGS.competition_data)
competition_num = saveImage("/media/ai/DL_DATA/HCCR/original_data/competition/", competition_writer, isAug=False)
competition_writer.close()
'''
print('train_num:', train_num)
#print('test_num:', test_num)
#print('competition_num:', competition_num)

