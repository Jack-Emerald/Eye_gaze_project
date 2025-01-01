import csv
import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from keras import Sequential
import keras
import os
from keras import layers
from keras import utils
from keras import Model
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


class GestureDataProcessor:
    def __init__(self):
        #self.feature_match = {"fixcenter": 1, "highdynamic": 2, "lowdynamic": 3, "speaker": 4, "relax": 5}
        self.feature_match = {"fixcenter": 1, "highdynamic": 2, "relax": 3}
        self.gesture_name = ["fixcenter", "highdynamic", "relax"]
        self.loaded_x = list()
        self.loaded_y = list()
        self.testFile = list()
        self.trainFile = list()
        self.stepLen = '32'
        self.totalAcc = float()

        self.folder_path = "all_gazes_text/"
        self.cross_validation()




    def cross_validation(self):
        # Get all possible combinations of 8 numbers from the list of 1 to 10
        self.trainFile = [[11,12,13,14,16,17,18,19,
                           21,22,23,24,26,27,28,29,
                           31,32,33,34,36,37,38,39,
                           41,42,43,44,46,47,48,49,
                           51,52,53,54,56,57,58,59,
                           61,62,63,64,66,67,68,69,
                           71,72,73,74,76,77,78,79,
                           81,82,83,84,86,87,88,89]]
        self.testFile = [[15, 110,
                          25, 210,
                          35, 310,
                          45, 410,
                          55, 510,
                          65, 610,
                          75, 710,
                          85, 810]]


    def generate_graph(self, file_path):
        # Initialize lists to store the data
        combined_list = list()
        sublists = list()

        # Read the file and parse the data
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():  # Ensure the line is not empty
                    # Split the line into its components
                    parts = line.split(",")
                    position_x = parts[2].replace("current eyeGaze_left: Position ", "").strip()
                    pos_x = float(position_x.replace("(", ""))
                    pos_y = float(parts[3])
                    pos_z = float(parts[4].replace(")", ""))

                    orientation_x = parts[5].replace("Orientation", "").strip()
                    ori_x = float(orientation_x.replace("(", ""))
                    ori_y = float(parts[6])
                    ori_z = float(parts[7])
                    ori_w = float(parts[8].replace(")", ""))

                    combined_list = combined_list + [ori_x, ori_y, ori_z, ori_w]

        #how many seconds does each step has
        step = 16*int(self.stepLen)

        # Split the combined_list into sublists
        for i in range(0, len(combined_list), step):
            time_step_data = combined_list[i:i + step]
            if len(time_step_data) == step:
                sublist = [time_step_data[i:i + 4] for i in range(0, len(time_step_data), 4)]
                sublists.append(sublist)

        return sublists
    def load_group(self, gesture_names, group, combination_index):
        if group == "train":
            range_domain = self.trainFile[combination_index]
        elif group == "test":
            range_domain = self.testFile[combination_index]

        for gesture_name in gesture_names:

            for i in range_domain:
                file_path = os.path.join(self.folder_path, f"{gesture_name}{i}.txt")
                data_list = self.generate_graph(file_path)

                for step in data_list:
                    self.loaded_y.append([self.feature_match[gesture_name]])

                self.loaded_x += data_list

        loaded_data_x = np.array(self.loaded_x)
        loaded_data_y = np.array(self.loaded_y)
        #print(loaded_data_x)
        #print(loaded_data_y)
        self.loaded_y = list()
        self.loaded_x = list()

        return loaded_data_x, loaded_data_y

    def load_dataSet(self, combination_index):
        trainX, trainY = self.load_group(self.gesture_name, "train", combination_index)
        print(trainX.shape, trainY.shape)
        testX, testY = self.load_group(self.gesture_name, "test", combination_index)
        print(testX.shape, testY.shape)

        # zero-offset class values
        trainY = trainY - 1
        testY = testY - 1
        # one hot encode y
        trainY = utils.to_categorical(trainY)
        testY = utils.to_categorical(testY)
        print(trainX.shape, trainY.shape, testX.shape, testY.shape)
        return trainX, trainY, testX, testY

    def evaluate_model(self, trainX, trainy, testX, testy):

        verbose, epochs, batch_size = 0, 30, 64
        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

        # Build the model using Sequential API
        inputs = layers.Input(shape = (n_timesteps, n_features))

        # LSTM layer with return_sequences=True to maintain the time steps for attention
        lstm_out = layers.LSTM(100, return_sequences = True)(inputs)

        # Attention mechanism: Apply self-attention (query and value are both LSTM outputs)
        attention_out = layers.Attention()([lstm_out, lstm_out])
        #attention_out = layers.MultiHeadAttention(num_heads=4, key_dim=100)(lstm_out, lstm_out)

        # Flatten the attention output to feed into Dense layers
        flattened_out = layers.Flatten()(attention_out)

        # Add the dense layer with 100 units and ReLU activation (same as original)
        dense_out = layers.Dense(100, activation = 'relu')(flattened_out)

        # Dropout layer for regularization
        dense_out = layers.Dropout(0.3)(dense_out)

        # Final dense layer with softmax activation (same as original)
        output = layers.Dense(n_outputs, activation = 'softmax')(dense_out)

        # Build the model
        model = Model(inputs = inputs, outputs = output)
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        # An epoch refers to one complete pass through the entire training dataset
        # Stop if model doesn’t improve on the validation set for 5 consecutive epochs
        early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5)
        model.fit(trainX, trainy, epochs, batch_size, verbose=verbose, validation_split=0.1, callbacks=[early_stopping])

        # Fit the model
        #model.fit(trainX, trainy, epochs = epochs, batch_size = batch_size, verbose = verbose)

        # Evaluate the model on test data
        _, accuracy = model.evaluate(testX, testy, batch_size = batch_size, verbose = 0)

        # Predict the test data and calculate the confusion matrix
        y_pred = model.predict(testX, batch_size = batch_size, verbose = 0)
        y_pred_labels = np.argmax(y_pred, axis = 1)
        y_true_labels = np.argmax(testy, axis = 1)

        # Print the confusion matrix
        conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
        print(conf_matrix / conf_matrix.astype("float").sum(axis=1))

        return accuracy, conf_matrix


    def summarize_results(self, scores):
        #print(scores)
        m, s = mean(scores), std(scores)
        Ave_acc = 'Accuracy: %.3f%% (+/-%.3f)' % (m, s)
        print(Ave_acc)

        return Ave_acc

    def to_csv(self, ave_acc, overall_conf_matrix):
        csv_file = 'Model_eval.csv'
        # Convert the confusion matrix to a string to save as one block
        conf_matrix_str = np.array2string(overall_conf_matrix, separator = ',',
                                          formatter = {'int': lambda x: "%d" % x})

        # Load the CSV file
        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            # If file doesn't exist, create one with a column named '64'
            df = pd.DataFrame(columns = [self.stepLen])

        # Find the first empty row in the '64' column
        first_empty_idx = df[self.stepLen].isna().idxmax()

        # If all rows are filled, append at the end of the file
        if first_empty_idx is None:
            print("append at end.")
            first_empty_idx = len(df)

        # Expand DataFrame if necessary
        if len(df) < first_empty_idx + 1:
            print("expand dataframe")
            # Expand DataFrame by adding one additional row
            df = pd.concat(
                [df, pd.DataFrame(np.nan, index = [first_empty_idx], columns = df.columns)],
                ignore_index = True)

        # Insert mean score and confusion matrix as a string into the row
        df.at[first_empty_idx, self.stepLen] = ave_acc
        df.at[first_empty_idx, self.stepLen] += conf_matrix_str

        # Save the updated DataFrame back to the CSV
        df.to_csv(csv_file, index = False)

    # run an experiment
    def run_experiment(self, repeats = 1):
        for index in range(len(self.trainFile)):
            print(index)
            trainX, trainy, testX, testy = self.load_dataSet(index)
            # repeat experiment
            scores = list()
            overall_conf_matrix = None

            for r in range(repeats):
                score, conf_matrix = self.evaluate_model(trainX, trainy, testX, testy)
                score = score * 100.0
                print('>#%d: %.3f' % (r + 1, score))
                scores.append(score)

                # Accumulate confusion matrices
                if overall_conf_matrix is None:
                    overall_conf_matrix = conf_matrix
                else:
                    overall_conf_matrix += conf_matrix

            # summarize results
            ave_acc = self.summarize_results(scores)
            self.totalAcc += mean(scores)
            print(f"Overall Confusion Matrix:\n{overall_conf_matrix}")

            # Plot heatmap for the overall confusion matrix
            plt.figure(figsize=(10, 7))
            sns.heatmap(overall_conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=True)
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.title("Overall Confusion Matrix Heatmap")
            plt.show()

            overall_conf_matrix = overall_conf_matrix / overall_conf_matrix.sum(axis = 1,
                                                                                           keepdims = True) * 100
            # Plot heatmap for the overall confusion matrix
            plt.figure(figsize=(10, 7))
            sns.heatmap(overall_conf_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True)
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.title("Overall Confusion Matrix Heatmap percentage")
            plt.show()

            #self.to_csv(ave_acc, overall_conf_matrix)

        print(f"Average acc is {self.totalAcc/len(self.trainFile)}%")



# Usage example:
processor = GestureDataProcessor()

processor.run_experiment()
