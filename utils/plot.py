import os
import matplotlib.pyplot as plt

# 无图形界面需要加，否则plt报错
plt.switch_backend("agg")


def loss_acc_plot(args, history):
    train_loss = history["train_loss"]
    eval_loss = history["eval_loss"]
    train_accuracy = history["train_acc"]
    eval_accuracy = history["eval_acc"]
    train_f1 = history["train_f1"]
    eval_f1 = history["eval_f1"]

    fig = plt.figure(figsize=(20, 20))
    fig.add_subplot(3, 1, 1)
    plt.title("loss during train")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    epochs = range(1, len(train_loss)+1)
    plt.plot(epochs, train_loss)
    plt.plot(epochs, eval_loss)
    plt.legend(["train_loss", "eval_loss"])

    fig.add_subplot(3, 1, 2)
    plt.title("accuracy during train")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_accuracy)
    plt.plot(epochs, eval_accuracy)
    plt.legend(["train_acc", "eval_acc"])

    fig.add_subplot(3, 1, 3)
    plt.title("f1 during train")
    plt.xlabel("epochs")
    plt.ylabel("f1")
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_f1)
    plt.plot(epochs, eval_f1)
    plt.legend(["train_f1", "eval_f1"])

    plt.savefig(os.path.join(args.image_dir, "loss_acc.png"))
