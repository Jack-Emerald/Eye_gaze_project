import csv
import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from keras import Sequential
import keras
from keras import layers
from keras import utils
from keras import Model
from keras import metrics
from keras import losses
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


class GestureDataProcessor:
    def __init__(self):
        self.feature_match = {"fixcenter": 1, "highdynamic": 2, "lowdynamic": 3, "speaker": 4, "relax": 5}
        self.gesture_name = ["fixcenter", "highdynamic", "lowdynamic", "speaker", "relax"]
        self.loaded_x = list()
        self.loaded_y = list()
        self.testFile = list()
        self.trainFile = list()
        self.stepLen = '32'
        self.totalAcc = float()

        self.cross_validation()




    def cross_validation(self):
        # Get all possible combinations of 8 numbers from the list of 1 to 10
        '''
        self.trainFile = [[11,12,13,14,16,17,18,19,
                           21,22,23,24,26,27,28,29,
                           31,32,33,34,36,37,38,39,
                           41,42,43,44,46,47,48,49]]
        self.testFile = [[15, 110,
                          25, 210,
                          35, 310,
                          45, 410]]
        '''
        self.trainFile = [[11,12,13,14,16,17,18,19]]
        self.testFile = [[15, 110]]


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
                file_path = f"{gesture_name}{i}.txt"
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

    # Define the Transformer encoder
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout = 0):
        x = layers.MultiHeadAttention(key_dim = head_size, num_heads = num_heads,
                                      dropout = dropout)(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon = 1e-6)(x)
        res = x + inputs

        x = layers.Conv1D(filters = ff_dim, kernel_size = 1, activation = "relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters = inputs.shape[-1], kernel_size = 1)(x)
        x = layers.LayerNormalization(epsilon = 1e-6)(x)
        return x + res

    # Build the model with stacked Transformer encoders
    def build_model(self, input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units,
                    dropout = 0, mlp_dropout = 0, n_classes = 8):
        inputs = layers.Input(shape = input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = layers.GlobalAveragePooling1D()(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation = "relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(n_classes, activation = "softmax")(x)

        model = keras.Model(inputs, outputs)
        return model

    def evaluate_model(self, trainX, trainy, testX, testy):
        verbose, epochs, batch_size = 0, 15, 64
        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
        print("In evaluate.")

        # Build the Transformer model
        model = self.build_model(
            input_shape = (n_timesteps, n_features),
            head_size = 256,
            num_heads = 4,
            ff_dim = 4,
            num_transformer_blocks = 4,
            mlp_units = [128],
            dropout = 0.25,
            mlp_dropout = 0.4,
            n_classes = n_outputs
        )

        print("In evaluate2.")

        model.compile(
            loss = "categorical_crossentropy",
            optimizer = keras.optimizers.Adam(learning_rate = 1e-4),
            metrics = ["accuracy"],
        )
        print("In evaluate3")

        # Train the model
        model.fit(trainX, trainy, epochs = epochs, batch_size = batch_size, verbose = verbose)

        print("In evaluate4.")

        # Evaluate and predict
        _, accuracy = model.evaluate(testX, testy, batch_size = batch_size, verbose = 0)

        print("In evaluate5.")
        y_pred = model.predict(testX, batch_size = batch_size, verbose = 0)
        y_pred_labels = np.argmax(y_pred, axis = 1)
        y_true_labels = np.argmax(testy, axis = 1)

        print("In evaluate6.")

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
        print(conf_matrix / conf_matrix.astype("float").sum(axis = 1))

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
    def run_experiment(self, repeats = 5):
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

            self.to_csv(ave_acc, overall_conf_matrix)

        print(f"Average acc is {self.totalAcc/len(self.trainFile)}%")



# Usage example:
processor = GestureDataProcessor()

processor.run_experiment()
