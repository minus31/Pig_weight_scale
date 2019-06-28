import numpy as np 
import pickle 

# load dataset 
IMAGE_PATH = '../sample_image/final_df.bin' 

def preprocess_input(image_path):
    with open(image_path, 'rb') as f :
        crop_img = pickle.load(f)
        
    X = crop_img.crop_images[np.newaxis, -1, -1 ,-1]
    y = crop_img.weight.values

    # train split 
    idx = list(np.random.choice(range(len(X)), int(len(X) * 0.8), replace=False))
    test_idx = list(set(list(range(len(X)))) - set(idx))
    X_train = X[idx]
    y_train = y[idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    return X_train, y_train, X_test, y_test

from keras.optimizers import Adam
from keras import losses
from network import DenseNet

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=1000)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--num_classes', type=int, default=600)
    args.add_argument('--input_shape', type=int, default=(256, 256, 3))
    args.add_argument('--sbow_shape', type=int, default=(128,))
    args.add_argument('--train', type=bool, default=False)
    args.add_argument('--updateDB', type=bool, default=False)
    args.add_argument('--eval', type=bool, default=False)
    args.add_argument('--model_path', type=str,
                      default="./checkpoint/finish")
    args.add_argument('--dataset_path', type=str, default="./data/images/")
    args.add_argument('--checkpoint_path', type=str, default="./checkpoint/")
    args.add_argument('--checkpoint_inteval', type=int, default=10)
    args.add_argument('--k', type=int, default=21)

    config = args.parse_args()


model = DenseNet()
op = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-10, decay=0.008, amsgrad=False)
model.compile(loss=losses.logcosh, optimizer=op, metrics=['mae'])

hist = model.fit(X_train, y_train, epochs=51, batch_size=24, validation_data=(X_test, y_test), verbose=2)