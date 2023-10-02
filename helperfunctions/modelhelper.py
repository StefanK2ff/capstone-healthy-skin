# helperfunctions.modelhelper.py

import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report

from .imagehelper import img_load_and_transform

import seaborn as sns
import matplotlib.pyplot as plt



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


def model_accuracy_on_test(model, test_df, targetvar, imagesize, verbose=2) -> tuple:
    """
    Compute the accuracy of the model on the test set

    Args:
        model (Model): Model to evaluate
        test_df (DataFrame): Test DataFrame
        targetvar (str): Target variable
        imagesize (tuple): Image size used to train the model
        verbose (int, optional): Verbosity. Defaults to 1 (print messages), 2 (print messages and plots) 0 (silent)    

    Returns:
        tuple: Test loss, test accuracy, confusion matrix, ROC AUC
    """

    # load test images from test_df
    test_images = []
    test_labels = []

    # load test images
    if verbose >= 1:
        print(f" > Evaluating model {model.name}:")
        print(" > Loading test images...")

    for i in range(test_df.shape[0]):
        image_path = test_df.iloc[i]['image_path']
        img = img_load_and_transform(image_path, imagesize)

        # convert img to np array 
        test_images.append(np.array(img))
        test_labels.append(test_df.iloc[i][targetvar])

    if verbose >= 1:
        print(" > Test images loaded.")
        print("\n")

    # using label encoder to get the labels
    if verbose >= 1:
        print(" > Converting labels to one hot format...")

    le = LabelEncoder()
    test_labels = le.fit_transform(test_df[targetvar])

    # converting to one hot format / categorial
    test_labels = to_categorical(test_labels)

    # reshaping the test images
    test_images = np.array(test_images).reshape(-1, imagesize[0], imagesize[1], 3)

    # Get model predictions
    if verbose >= 1:
        print(" > Getting model predictions...")

    predictions = model.predict(test_images)

    if verbose >= 1:
        print(" > Model predictions obtained.")

    # If it's binary classification, compute ROC AUC
    if predictions.shape[1] == 2:
        roc_auc = roc_auc_score(test_labels[:, 1], predictions[:, 1])

        if verbose >= 1:
            print(f" > ROC AUC: {roc_auc}")

    else:
        macro_auc = roc_auc_score(test_labels, predictions, multi_class='ovr', average='macro')
        micro_auc = roc_auc_score(test_labels, predictions, multi_class='ovr', average='micro')
        roc_auc = (micro_auc, macro_auc)

        if verbose >= 1:
            print(f" > Macro AUC: {macro_auc}")
            print(f" > Micro AUC: {micro_auc}")

    # Get the class with highest probability for each sample
    pred_labels = np.argmax(predictions, axis=1)

    # Convert one-hot encoded labels back to label-encoded
    true_labels = np.argmax(test_labels, axis=1)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    if verbose == 2:
        plt.figure(figsize=(10,7))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()
        print("\n")

    # Compute classification report
    if verbose >= 1:
        print(" > Computing classification report...")
        print(classification_report(true_labels, pred_labels))


    # Evaluate the model on the test set
    if verbose >= 1:
        print(" > Evaluating model on test set...")
          
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0 if verbose == 0 else 1)

    if verbose >= 1:
        print(" > Done.")

    return (test_loss, test_acc, conf_matrix, roc_auc)