import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def sample_nn1():
    # Make fake data
    X = np.arange(-100, 100, 4)
    y = np.arange(-90, 110, 4)
    # y = X + 10

    # Split the dataset to train/test/valid set
    X_train = X[:40]
    y_train = y[:40]
    X_test = X[40:]
    y_test = y[40:]

    # Visualize the data
    # plt.scatter(X_train, y_train, c="red", label="training data")
    # plt.scatter(X_test, y_test, c="green", label="testing data")
    # plt.legend()
    # plt.show()

    tf.random.set_seed(42)
    # Make model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=[1])
    ])
    # Compile model
    model.compile(
        loss=tf.keras.losses.mae,
        optimizer=tf.keras.optimizers.SGD(),
        metrics=["mae"]
    )
    # Fit the model
    model.fit(
        X_train,
        y_train,
        epochs=100
    )

    y_pred = model.predict(X_test)

    # Evaluate
    err = tf.metrics.mean_squared_error(
        y_true=y_test,
        y_pred=y_pred.squeeze()
    )
    print(f"Mean Squared error: {err}")

    plt.scatter(X_train, y_train, c="green", label="Training data")
    plt.scatter(X_test, y_test, c="red", label="Testing data")
    plt.scatter(X_test, y_pred, c="blue", label="Prediction data")
    plt.legend()
    plt.show()

def sample_nn2():


if __name__ == '__main__':
    sample_nn1()
