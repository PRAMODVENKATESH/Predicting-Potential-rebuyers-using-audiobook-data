# Predicting-Potential-rebuyers-using-audiobook-data
This ML algorithm takes 10 potential inputs from audiobook buyers such as metrics and predicts the customer behaviour (predicting whether an individual is a potential re-buyer).
The 10 metrics are chosen carefully eliminating any redundancies in inputs and combing the metrics implicitly wherever possible, thereby reducing the number of inputs.
The model inputs are loaded in a CSV format file, with the train, test , target variables are pre-processed to eliminate the imbalance in the datasets.
The next stage is loading the pre processed data into the model as a numpy array, with input and target variables as two separate arrays.
