import numpy
import pandas
import matplotlib.pyplot as plt


def my_plot(y_hat: numpy.ndarray, y: numpy.ndarray, start: int, end: int):
    range1 = numpy.arange(start, end)
    temp_y = y.flatten()
    temp_y_hat = y_hat.flatten()
    fig, axs = plt.subplots(3)
    axs[0].plot(range1, temp_y, color='r', label="y")
    axs[0].set_title("Actual values")
    axs[1].plot(range1, temp_y_hat, color='b', label="y_hat")
    axs[1].set_title("Predicted values")
    axs[2].plot(range1, temp_y, color='r', label="y")
    axs[2].plot(range1, temp_y_hat, color='b', label="y_hat")
    axs[2].set_title("Actual values and Predicted values")
    plt.show()


def regression(x: numpy.ndarray, y: numpy.ndarray):
    w = numpy.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
    y_hat = x.dot(w)
    l2_norm = 0
    for i in range(len(y_hat)):
        l2_norm += (y_hat[i][0] - y[i][0])**2
    print("w :\n", w)
    print("l2 norm of regression: ", l2_norm)
    return y_hat, l2_norm, w


def predict(file: pandas.core.frame.DataFrame, w: numpy.ndarray, window: int):
    y = numpy.array([])
    samples = len(file.values)
    for i in range(samples):
        y = numpy.append(y, file.values[i][2])  # append the closing price.
    y = y.reshape(samples, 1)
    y_predicted = numpy.array([])
    y_copy = numpy.copy(y)
    for i in range(samples//2, samples):
        x = numpy.array([])
        x = numpy.append(x, 1)  # coefficient of w0.
        for j in range(window, 0, -1):
            x = numpy.append(x, y_copy[i - j][0])
        x = x.reshape(1, window + 1)
        y_copy[i] = x.dot(w)
        y_predicted = numpy.append(y_predicted, y_copy[i])
    y_predicted = y_predicted.reshape(len(y_predicted), 1)
    y = numpy.delete(y, numpy.s_[:samples//2], axis=0)
    l2_norm = 0
    for i in range(len(y)):
        l2_norm += (y[i][0] - y_predicted[i][0]) ** 2
    print("l2 norm of prediction: ", l2_norm)
    my_plot(y_predicted, y, samples//2, samples)


def main():
    file = pandas.read_csv("BTC_USD_2020-10-31_2021-01-30-CoinDesk.csv")
    y = numpy.array([])
    samples = len(file.values)//2
    window = 20
    for i in range(samples):
        y = numpy.append(y, file.values[i][2])              # append the closing price.
    y = y.reshape(samples, 1)
    x = numpy.array([])
    for i in range(samples-window):
        x = numpy.append(x, 1)                              # coefficient of w0.
        for j in range(window):
            x = numpy.append(x, y[i + j][0])
    x = x.reshape(samples-window, window+1)
    y = numpy.delete(y, numpy.s_[:window], axis=0)
    y_hat, l2_norm, w = regression(x, y)
    predict(file, w, window)


if __name__ == "__main__":
    main()