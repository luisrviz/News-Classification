from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


def save_plot_history(history):
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('results/nn_learning.png')


def save_classification_report(df_test_y, df_test_y_pred, name):
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    c_r = classification_report(df_test_y, df_test_y_pred,
                          target_names=classes, output_dict=True)
    df = pd.DataFrame(c_r).transpose()
    df.to_csv(f"results/{name}_clasification_report.csv")


def save_confusion_matrix(df_test_y, df_test_y_pred, name):
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    cm = confusion_matrix(df_test_y, df_test_y_pred)
    df_cfm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(15, 10))
    cfm_plot = sn.heatmap(df_cfm, annot=True)
    cfm_plot.figure.savefig(f"results/{name}_cfm.png")
