# load Dataset

from tensorflow.keras.models import load_model
import numpy as np 
import pickle 

# load dataset 
# with open('../sample_image/sample560_crop_df.bin', 'rb') as f :
#     crop_img = pickle.load(f)
    
model = load_model('../model_hdf5/DenseNet_0720_50Epochs.hdf5')

def scaler(img):

    weigh = model.predict(img)
    print(weigh)
    return weigh


if __name__ == "__main__":
    pass

