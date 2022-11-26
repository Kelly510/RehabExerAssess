import os
import matplotlib.pyplot as plt


def plot_train_curve(losses_epoch, accuracy_epoch, recall_epoch,
                     precision_epoch, fig_dir):
    plt.figure(figsize=[15, 12], dpi=200)

    plt.subplot(2, 2, 1)
    plt.plot(losses_epoch)
    plt.title('Loss')

    plt.subplot(2, 2, 2)
    plt.plot(accuracy_epoch)
    plt.title('Accuracy')
    plt.ylim([0, 1])

    plt.subplot(2, 2, 3)
    plt.plot(recall_epoch)
    plt.title('Recall')
    plt.ylim([0, 1])

    plt.subplot(2, 2, 4)
    plt.plot(precision_epoch)
    plt.title('Precision')
    plt.ylim([0, 1])

    plt.savefig(os.path.join(fig_dir, 'train_curve.jpg'))

    plt.clf()
    plt.close('all')
