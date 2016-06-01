import matplotlib.pyplot as plt
import numpy as np
import data

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def visualize_data(data, padsize=1, padval=0, cmap="gray", image_size=(10,10)):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    #plt.figure(figsize=image_size)
    plt.imshow(data, cmap=cmap)
    plt.show()
    plt.axis('off')

if __name__ == "__main__":
    train_X,_,_,_,_,_,_ = data.load()

    random_idxs = np.random.randint(0,train_X.shape[0], 16)
    visualize_data(train_X[random_idxs].transpose(0,2,3,1))
