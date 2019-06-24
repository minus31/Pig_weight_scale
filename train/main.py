import numpy as np 
import pickle 

# load dataset 
IMAGE_PATH = '../sample_image/final_df.bin' 
with open(IMAGE_PATH, 'rb') as f :
    crop_img = pickle.load(f)
    
X = np.array([i for i in crop_img.crop_images])
X = crop_img.crop_images[np.newaxis, -1, -1 ,-1]
y = crop_img.weight.values

idx = list(np.random.choice(range(len(X)), int(len(X) * 0.8), replace=False))
test_idx = list(set(list(range(len(X)))) - set(idx))

X_train = X[idx]
y_train = y[idx]

X_test = X[test_idx]
y_test = y[test_idx]