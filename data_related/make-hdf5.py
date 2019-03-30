# from PIL import Image
import numpy as np
import csv
import h5py

IMAGE_DIR = './image/'
CSV_FILE = './annotation.csv'
HDF5_FILE = './mydata.h5'
TEST_HDF5_FILE = './testdata.h5'

image_lists = []
label_lists = []
test_image_list = []
test_label_list = []

# print('please wait...')
# img_num = len(f.readlines()) #2370
# test_index = round(img_num*0.9) #2133
# print(test_index)
index = 0

with open(CSV_FILE, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if index < 1896:
            image_lists.append(row[0] + '.png')
            label = (row[1],row[2],row[3],row[4])
            label_lists.append(label)
        else:
            test_image_list.append(row[0] + '.png')
            label = (row[1],row[2],row[3],row[4])
            test_label_list.append(label)
        index = index + 1


datas = np.zeros((len(image_lists),3,200,200))
labels = np.zeros((len(image_lists), 4))
test_datas = np.zeros((len(test_image_list),3,200,200))
test_labels = np.zeros((len(test_image_list), 4))

for ii, _file in enumerate(image_lists):
    img = Image.open(IMAGE_DIR + _file).resize((200,200),Image.ANTIALIAS)
    img_data = np.array(img).transpose((2,0,1))
    datas[ii, :, :, :] = img_data.astype(np.float32)
    labels[ii, :] = np.array(label_lists[ii]).astype(np.int)

for ii, _file in enumerate(test_image_list):
    img = Image.open(IMAGE_DIR + _file).resize((200,200),Image.ANTIALIAS)
    img_data = np.array(img).transpose((2,0,1))
    test_datas[ii, :, :, :] = img_data.astype(np.float32)
    test_labels[ii, :] = np.array(test_label_list[ii]).astype(np.int)
    
print(datas.shape)
print(labels.shape)
print(test_datas.shape)
print(test_labels.shape)

with h5py.File(HDF5_FILE, 'w') as f:
    f['data'] = datas
    f['labels'] = labels
    f.close()

with h5py.File(TEST_HDF5_FILE, 'w') as f:
    f['data'] = test_datas
    f['labels'] = test_labels
    f.close()

print('done...')
