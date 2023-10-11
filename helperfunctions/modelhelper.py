# helperfunctions.modelhelper.py

import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
import tensorflow as tf

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


def model_accuracy_on_test(model,  test_df, targetvar, imagesize, verbose=2) -> tuple:
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
        test_images.append(np.array(img)/255.0)
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
          
    results = model.evaluate(test_images, test_labels, verbose=0 if verbose == 0 else 1)
    return_results = []

    for metric_value, metric_name in zip(results, model.metrics_names):
        print(f"{metric_name}: {metric_value}")
        return_results.append({metric_name: metric_value})

    if verbose >= 1:
        print(" > Done.")

    return (return_results, conf_matrix, roc_auc)

def poly1_cross_entropy(logits, labels, epsilon=-1.0):
    """
    PolyLoss function with Poly1 variation.
    
    Parameters:
    - logits: The network's output logits.
    - labels: True labels.
    - epsilon: A parameter for the Poly1 loss. Default is -1.0.
    
    Returns:
    - Poly1 loss.
    """
    
    # pt has shape [batch]
    pt = tf.reduce_sum(labels * tf.nn.softmax(logits, axis=-1), axis=-1)
    
    # Compute softmax cross-entropy
    CE = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    
    # Compute Poly1
    Poly1 = CE + epsilon * (1 - pt)
    
    return tf.reduce_mean(Poly1)


def focal_loss_binary(alpha=0.25, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

def focal_loss_multiclass(alpha, gamma=2.0):
    alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)
    def multi_class_focal_loss(y_true, y_pred):
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return multi_class_focal_loss

def f1_score(y_true, y_pred):
    # Calculate precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())

    # Calculate recall
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())

    # Calculate F1 score
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
