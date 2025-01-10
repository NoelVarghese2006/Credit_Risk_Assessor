#extra layer

import tensorflow as tf
import datetime
import numpy as np
import matplotlib.pyplot as plt

# Make a bigger dataset
X = np.arange(-100, 100, 4)
# Make labels for the dataset (adhering to the same pattern as before)
y = np.arange(-90, 110, 4)
# Split data into train and test sets
X_train = X[:40] # first 40 examples (80% of data)
y_train = y[:40]

X_test = X[40:] # last 10 examples (20% of data)
y_test = y[40:]

# plt.figure(figsize=(10, 7))
# # Plot training data in blue
# plt.scatter(X_train, y_train, c='b', label='Training data')
# # Plot test data in green
# plt.scatter(X_test, y_test, c='g', label='Testing data')
# # Show the legend
# plt.legend()

# Set random seed
tf.random.set_seed(42)

# Create a model (same as above)
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1, input_shape=[1]),
  tf.keras.layers.Dense(1, input_shape=[1])
])

# Compile model (same as above)
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

# Fit model (same as above)
model.fit(X_train, y_train, epochs=100) # commented out on purpose (not fitting it just yet)

# Make predictions
y_preds = model.predict(X_test)

def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=y_preds):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))
  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", label="Training data")
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", label="Testing data")
  # Plot the predictions in red (predictions were made on the test data)
  plt.scatter(test_data, predictions, c="r", label="Predictions")
  # Show the legend
  plt.legend()
  plt.show()

plot_predictions(train_data=X_train,
                 train_labels=y_train,
                 test_data=X_test,
                 test_labels=y_test,
                 predictions=y_preds)


model.evaluate(X_test, y_test)
# # Calculate the mean absolute error
# mae = tf.keras.metrics.MeanAbsoluteError()
# mae.update_state(y_true=y_test, y_pred=y_preds)
# print(mae.result().numpy())