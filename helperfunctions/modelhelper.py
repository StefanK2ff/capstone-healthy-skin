# helperfunctions.modelhelper.py

import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from .imagehelper import img_load_and_transform



def model_plot_accuracy(history) -> None:
    """
    Plot the accuracy of the model
    Args:
        history (History): History object of the model
    """

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')


def model_accuracy_on_test(model, test_df, targetvar, imagesize=(128,128)) -> None:

    # load test images from test_df
    test_images = []
    test_labels = []

    for i in range(test_df.shape[0]):
        image_path = test_df.iloc[i]['image_path']
        img = img_load_and_transform(image_path)

        # convert img to np array 
        test_images.append(np.array(img))
        test_labels.append(test_df.iloc[i][targetvar])

    # using label encoder to get the labels
    le = LabelEncoder()
    test_labels = le.fit_transform(test_df[targetvar])

    # converting to one hot format / categorial
    test_labels = to_categorical(test_labels)

    # reshaping the test images
    test_images = np.array(test_images).reshape(-1, imagesize[0], imagesize[1], 3)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)