import time
import pickle
import numpy as np
import random
import os
import sys

'''
Note: final pickle pulls the 'spectrum' key of all
 the `.1sec_hdpickles` and aggregates them
'''
# read all the pickles

def data_aggre(files, labels):
    '''
    futher cleanses
    checks if dim = 66
    '''
    nv_shape = (201, 66)
    features = []
    f_labels = []

    cnt = 0
    for i in range(files.shape[0]):
        file = files[i]
        f_label = labels[i]
        with open(file, 'rb') as jar:
            data_file = pickle.load(jar)

        #print(data_file['spectrum'].shape)
        if(data_file['spectrum'].shape[1] == 66):
            cnt +=1
            feature = data_file['spectrum'].reshape(nv_shape)
            features.append(feature)
            f_labels.append(f_label)

    features_shape = (cnt, 201, 66, 1)

    features = np.array(features).reshape(features_shape)
    f_labels = np.array(f_labels)
    return features, f_labels

# new 1 sec pickles
librispeech_spectrogram_path = os.path.join("/Users/vikuo/Documents/accent-classification-corpora/librispeech_1sec_hdpickle")
librit_spectrogram_path =os.path.join("/Users/vikuo/Documents/accent-classification-corpora/librit_1sec_hdpickle")

folders = [librispeech_spectrogram_path, librit_spectrogram_path]
train_files = []
val_files = []

train_frac = 0.9

for file_path in folders:
    file_list = np.array(os.listdir(file_path))
    no_of_files = len(file_list)
    indicies = np.array(np.random.choice(no_of_files,no_of_files, replace=False))
    no_of_train_files = int(train_frac*no_of_files)
    train_files.append(np.array([file_path+'/'+i for i in file_list[indicies[0:no_of_train_files]]]))
    val_files.append(np.array([file_path+'/'+ i for i in file_list[indicies[no_of_train_files:]]]))
    print(file_path,' - train_files = ',no_of_train_files,', val files = ',no_of_files - no_of_train_files)

# shuffle indices for trian and valide
train_all_files = np.concatenate((train_files[0],train_files[1]))
train_all_labels = np.concatenate((np.zeros(len(train_files[0])),np.ones(len(train_files[1]))))
len_train = len(train_all_labels)
indicies = np.array(np.random.choice(len_train,len_train, replace=False))
train_all_files = train_all_files[indicies]
train_all_labels = train_all_labels[indicies]

val_all_files = np.concatenate((val_files[0],val_files[1]))
val_all_labels = np.concatenate((np.zeros(len(val_files[0])),np.ones(len(val_files[1]))))
len_val = len(val_all_labels)
indicies = np.array(np.random.choice(len_val,len_val, replace=False))
val_all_files = val_all_files[indicies]
val_all_labels = val_all_labels[indicies]

print("val_all_labels", val_all_labels.shape)




# aggregate training

start_time = time.time()
print(train_all_files.shape)
x_train, y_train = data_aggre(train_all_files, train_all_labels)
print("--- %s seconds ---" % (time.time() - start_time))
print(x_train.shape)


# pickle all the training, extract form the specto dictionary
max_bytes = 2**31 - 1

file_path = "/Users/vikuo/Documents/accent-classification-corpora/cnn_proj_yvg_testing/train_1_sec_hd_feature.pickle"
bytes_out = pickle.dumps(x_train)
with open(file_path, 'wb') as f_out:
    for idx in range(0, len(bytes_out), max_bytes):
        f_out.write(bytes_out[idx:idx+max_bytes])

print("################################## \
 \n# test loading piclles, verify shape \n \
#################################")
##################################
# test loading piclles, verify shape
#################################
file_path = "/Users/vikuo/Documents/accent-classification-corpora/cnn_proj_yvg_testing/train_1_sec_hd_feature.pickle"
## read
bytes_in = bytearray(0)
input_size = os.path.getsize(file_path)
with open(file_path, 'rb') as f_in:
    for _ in range(0, input_size, max_bytes):
        bytes_in += f_in.read(max_bytes)
x_train_2 = pickle.loads(bytes_in)

print(x_train_2.shape)


file_path = "/Users/vikuo/Documents/accent-classification-corpora/cnn_proj_yvg_testing/train_1_sec_hd_label.pickle"
bytes_out = pickle.dumps(y_train)
with open(file_path, 'wb') as f_out:
    for idx in range(0, len(bytes_out), max_bytes):
        f_out.write(bytes_out[idx:idx+max_bytes])


# pickle all the validation, extract form the specto dictionary
start_time = time.time()
print(val_all_files.shape)
x, y = data_aggre(val_all_files, val_all_labels)
print("--- %s seconds ---" % (time.time() - start_time))
print(x.shape)
print(y.shape)
# this is redundent, i carries through my typo in the prev cell
# shouldve made a function and rename to val and train in the end but whatev
x_val = x
y_val = y


file_path = "/Users/vikuo/Documents/accent-classification-corpora/cnn_proj_yvg_testing/val_1_sec_hd_feature.pickle"
bytes_out = pickle.dumps(x_val)
with open(file_path, 'wb') as f_out:
    for idx in range(0, len(bytes_out), max_bytes):
        f_out.write(bytes_out[idx:idx+max_bytes])

file_path = "/Users/vikuo/Documents/accent-classification-corpora/cnn_proj_yvg_testing/val_1_sec_hd_label.pickle"
bytes_out = pickle.dumps(y_val)
with open(file_path, 'wb') as f_out:
    for idx in range(0, len(bytes_out), max_bytes):
        f_out.write(bytes_out[idx:idx+max_bytes])
