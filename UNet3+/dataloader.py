import numpy as np
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import os
import random

def extract_bayer_channels(raw):
    # Reshape the input bayer image
    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)

    return RAW_norm

class ZurichSequence(keras.utils.Sequence):

    def __init__(self, 
                 data_dir,
                 train=True,
                 batch_size=5,  
                 input_size = (224,224,4),
                 target_size = (448,448,3),
                 shuffle=True,
                 seed = 101):

        if train:
            data_dir = os.path.join(data_dir,"train")
        else:
            data_dir = os.path.join(data_dir,"test")

        self.target_dir= os.path.join(data_dir,"canon")
        self.input_dir= os.path.join(data_dir,"huawei_raw")

        self.len = len(os.listdir(self.input_dir))
        self.num_batches = int(np.floor(self.len / batch_size))
        self.batch_size = batch_size
        self.indexes = list(range(self.len))
        self.num_images = self.len
        if shuffle:
            random.shuffle(self.indexes)
        self.input_size = input_size
        self.target_size = target_size

    def __len__(self):
        return self.num_batches
        
    def shuffle(self):
        random.shuffle(self.indexes)

    def get_image(self, idx):

        input_path = os.path.join(self.input_dir,str(idx)+".png")
        target_path = os.path.join(self.target_dir,str(idx)+".jpg")

        input = np.asarray(load_img(input_path, color_mode="grayscale"))/255
        input = extract_bayer_channels(input)

        target = np.asarray(load_img(target_path))
        target = target.astype(np.float32) / 255

        # Sample construction
        sample = (input, target)

        return sample

    def __getitem__(self, idx):

        img_idx = idx*self.batch_size
        indexes = self.indexes[img_idx : img_idx + self.batch_size]

        batch_input = np.zeros((self.batch_size, self.input_size[0], self.input_size[1], 4))
        batch_target = np.zeros((self.batch_size, self.target_size[0], self.target_size[1], 3))

        for i,idx in enumerate(indexes):
            input, target = self.get_image(idx)
            batch_input[i,:] = input
            batch_target[i, :] = target

        return batch_input, batch_target
        

'''Example
train_dataset = ZurichSequence(...)
valid_dataset = ZurichSequence(...)
# Train model on dataset
model.fit_generator(train_dataset, validation_data=valid_dataset)
'''
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_dataset = ZurichSequence(data_dir= "/storage/zurich-dataset")

    input, target = train_dataset[5]
    print(input.shape, target.shape)

    plt.imshow(target[0])
    plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
