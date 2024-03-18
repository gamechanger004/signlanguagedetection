from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt

# Define a function to plot a confusion matrix
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('confusion_matrix.png')

# Load a pre-trained Keras model for image classification
model = load_model('model/keypoint_classifier/keypoint_classifier_new.h5')

pred_labels = []
start_time = time.time()

# Make predictions on the test data using the loaded model
pred_probabs = model.predict(X_test)

end_time = time.time()
pred_time = end_time - start_time
avg_pred_time = pred_time / X_test.shape[0]

# Print the average prediction time per sample
print('Average prediction time: %fs' % (avg_pred_time))

# Determine the predicted labels based on maximum probabilities
for pred_probab in pred_probabs:
    pred_labels.append(list(pred_probab).index(max(pred_probab)))

# Compute the confusion matrix to evaluate model performance
cm = confusion_matrix(y_test, np.array(pred_labels))

# Generate and print the classification report
classification_report = classification_report(y_test, np.array(pred_labels))
print('\n\nClassification Report')
print('---------------------------')
print(classification_report)

# Plot and display the confusion matrix
plot_confusion_matrix(cm, range(44), normalize=False)
