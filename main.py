import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from NeuralNetwork import NeuralNetwork


def main():
    solve_classification()
    solve_regression()


def solve_classification():
    print("\n=== Classification problem: Student Dropout ===")
    df = pd.read_csv("student_dropout_dataset.csv")

    # Data preprocessing
    df.drop('student_id', axis=1, inplace=True)
    df.drop('region', axis=1, inplace=True)
    df.drop('enroll_date', axis=1, inplace=True)
    df.drop('label_name', axis=1, inplace=True)
    df.drop('label_multiclass', axis=1, inplace=True)

    corr = df.corr()
    cor_target = abs(corr["label"])
    relevant_features = cor_target[cor_target > 0.2]
    names = relevant_features.index.tolist()

    # Drop the target variable from the results
    names.remove('label')

    X = df[names].values
    y = df['label'].values

    # Standardization
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Basic model
    results = train_evaluate_model(X_train, y_train, X_test, y_test, "basic", 1)
    results.index = ["Model basic"]
    print(results)

    # 1. Parameter testing: Learning Rate
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    learning_results = []
    for i, lr in enumerate(learning_rates, start=1):
        results = train_evaluate_model(X_train, y_train, X_test, y_test, "Learning Rate", lr, learning_rate=lr)
        results.index = ["Model " + str(i)]
        learning_results.append(results)
    print(pd.concat(learning_results, ignore_index=False))

    # 2. Parameter testing: Epochs
    epochs_arr = [10, 100, 1000, 5000]
    epochs_results = []
    for i, epch in enumerate(epochs_arr, start=1):
        results = train_evaluate_model(X_train, y_train, X_test, y_test, "Epochs", epch, epochs=epch)
        results.index = ["Model " + str(i)]
        epochs_results.append(results)
    print(pd.concat(epochs_results, ignore_index=False))

    # 3. Parameter testing: Activation function
    act_functions = ["relu", "sigmoid", "linear"]
    act_results = []
    for i, function in enumerate(act_functions, start=1):
        results = train_evaluate_model(X_train, y_train, X_test, y_test, "Activation Function", function, activation=function)
        results.index = ["Model " + str(i)]
        act_results.append(results)
    print(pd.concat(act_results, ignore_index=False))

    # 4. Parameter testing: Number of neurons in hidden layer
    neurons = [2, 8, 16, 64]
    neurons_results = []
    for i, neuron in enumerate(neurons, start=1):
        results = train_evaluate_model(X_train, y_train, X_test, y_test, "Number of Neurons", neuron, layer_dimensions=[5, neuron, 1])
        results.index = ["Model " + str(i)]
        neurons_results.append(results)
    print(pd.concat(neurons_results, ignore_index=False))

    # 5. Parameter testing: Number of Layers
    layers = [[5, 16, 1], [5, 8, 8, 1], [5, 4, 4, 8, 1], [5, 4, 4, 4, 4, 1]]
    layers_results = []
    for i, l in enumerate(layers, start=1):
        results = train_evaluate_model(X_train, y_train, X_test, y_test, "Number of Layers", l, layer_dimensions=l)
        results.index = ["Model " + str(i)]
        layers_results.append(results)
    print(pd.concat(layers_results, ignore_index=False))

    # 6. Parameter testing: Testing split
    splits = [0.1, 0.2, 0.4, 0.8]
    splits_results = []
    for i, split_size in enumerate(splits, start=1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)
        results = train_evaluate_model(X_train, y_train, X_test, y_test, "Test Size", split_size, layer_dimensions=[5, 16, 1])
        results.index = ["Model " + str(i)]
        splits_results.append(results)

    final_results = pd.concat(splits_results, ignore_index=False)
    print(final_results)

    # 7. Parameter testing: weights initiation methods
    init_methods = ["he", "xavier", "small_random", "zeros"]
    init_results = []
    for i, method in enumerate(init_methods, start=1):
        results = train_evaluate_model(X_train, y_train, X_test, y_test, "Weight initiation methods", method, initiation_method=method)
        results.index = ["Model " + str(i)]
        init_results.append(results)

    final_results = pd.concat(init_results, ignore_index=False)
    print(final_results)

    # 8. Parameter testing: Momentum
    beta_values = [0.0, 0.5, 0.9, 0.99]
    beta_results = []
    for i, b in enumerate(beta_values):
        results = train_evaluate_model(X_train, y_train, X_test, y_test,"Momentum (Beta)", b, beta=b)
        results.index = ["Model " + str(i)]
        beta_results.append(results)

    final_results = pd.concat(beta_results, ignore_index=False)
    print(final_results)



def solve_regression():
    print("\n=== PROBLEM REGRESJI: Exam Performance ===")
    df = pd.read_csv("StudentPerformanceFactors.csv")
    df.dropna(inplace=True)

    # Selection of numerical data
    features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']
    X = StandardScaler().fit_transform(df[features].values)
    y = df['Exam_Score'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    layer_dimensions = [6, 16, 1]

    # Basic model
    results = train_evaluate_model(X_train, y_train, X_test, y_test, "basic", 1, layer_dimensions=layer_dimensions, problem_type="regression")
    results.index = ["Model basic"]
    print(results)

    # 1. Parameter testing: learning rate
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    learning_results = []
    for i, lr in enumerate(learning_rates, start=1):
        results = train_evaluate_model(X_train, y_train, X_test, y_test, "Learning Rate", lr, layer_dimensions=layer_dimensions, learning_rate=lr, problem_type="regression")
        results.index = ["Model " + str(i)]
        learning_results.append(results)
    print(pd.concat(learning_results, ignore_index=False))

    # 2. Parameter testing: Epochs
    epochs_arr = [10, 100, 1000, 5000]
    epochs_results = []
    for i, epch in enumerate(epochs_arr, start=1):
        results = train_evaluate_model(X_train, y_train, X_test, y_test, "Epochs", epch, layer_dimensions=layer_dimensions, epochs=epch, problem_type="regression")
        results.index = ["Model " + str(i)]
        epochs_results.append(results)
    print(pd.concat(epochs_results, ignore_index=False))

    # 3. Parameter testing: Activation function
    act_functions = ["relu", "sigmoid", "linear"]
    activation_results = []
    for i, function in enumerate(act_functions, start=1):
        results = train_evaluate_model(X_train, y_train, X_test, y_test, "Activation Function", function, layer_dimensions=layer_dimensions, activation=function, problem_type="regression")
        results.index = ["Model " + str(i)]
        activation_results.append(results)
    print(pd.concat(activation_results, ignore_index=False))

    # 4. Parameter testing: Number of neurons in hidden layer
    neuron_configs = [4, 8, 16, 32]
    neuron_results = []
    for i, nc in enumerate(neuron_configs, start=1):
        results = train_evaluate_model(X_train, y_train, X_test, y_test, "Neurons", nc, layer_dimensions=[6, nc, 1], problem_type="regression")
        results.index = ['Model ' + str(i)]
        neuron_results.append(results)
    print(pd.concat(neuron_results, ignore_index=False))

    # 5. Parameter testing: Number of Layers
    layers = [[6, 16, 1], [6, 8, 8, 1], [6, 4, 4, 8, 1], [6, 4, 4, 4, 4, 1]]
    layers_results = []
    for i, l in enumerate(layers, start=1):
        results = train_evaluate_model(X_train, y_train, X_test, y_test, "Number of Layers", l, layer_dimensions=l, problem_type="regression")
        results.index = ["Model " + str(i)]
        layers_results.append(results)
    print(pd.concat(layers_results, ignore_index=False))

    # 6. Parameter testing: Testing split
    splits = [0.1, 0.2, 0.4, 0.8]
    split_results = []
    for i, split_size in enumerate(splits, start=1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)
        y_train = scaler_y.fit_transform(y_train)
        y_test = scaler_y.transform(y_test)
        results = train_evaluate_model(X_train, y_train, X_test, y_test, "Test Size", split_size, layer_dimensions=layer_dimensions, problem_type="regression")
        results.index = ["Model " + str(i)]
        split_results.append(results)
    print(pd.concat(split_results, ignore_index=False))

    # 7. Parameter testing: weights initiation methods
    init_methods = ["he", "xavier", "small_random", "zeros"]
    method_results = []
    for i, method in enumerate(init_methods, start=1):
        results = train_evaluate_model(X_train, y_train, X_test, y_test, "Weight initiation methods", method, layer_dimensions=layer_dimensions, initiation_method=method, problem_type="regression")
        results.index = ["Model " + str(i)]
        method_results.append(results)
    print(pd.concat(method_results, ignore_index=False))

    # 8. Parameter testing: Momentum
    beta_values = [0.0, 0.5, 0.9, 0.99]
    beta_results = []
    for i, b in enumerate(beta_values):
        results = train_evaluate_model(X_train, y_train, X_test, y_test,"Momentum (Beta)", b, layer_dimensions, problem_type="regression", beta=b)
        results.index = ["Model " + str(i)]
        beta_results.append(results)
    print(pd.concat(beta_results, ignore_index=False))


def train_evaluate_model(X_train, y_train, X_test, y_test, tested_param_name, tested_param_value, layer_dimensions=None, learning_rate=0.1, problem_type="classification", epochs=1000, activation="relu", initiation_method="he", beta=0.0):
    """
    Keyword arguments:
    X_train -- Training data
    y_train -- Traing labels
    X_test -- test data
    y_test -- test labels
    layer_dimensions -- python array (list) containing the dimensions of each layer in our network
    problem_type -- 'classification' or 'regression'
    learning_rate -- learning rate of the network.
    epochs -- number of iterations of the optimization loop
    activation -- activation function for hidden layers
    tested_param_name -- name of the parameter being tested (e.g., "Learning Rate", "Activation")
    tested_param_value -- value of the parameter being tested
    returns a dataframe
    """
    if layer_dimensions is None:
        layer_dimensions = [5, 16, 1]
    temp_results = []
    for i in range(3): # repeating 3 times to get average result
        # create model instance with the given hyperparameters
        model = NeuralNetwork(learning_rate=learning_rate, layer_dimensions=layer_dimensions, problem_type=problem_type, activation_function=activation, initialization_method=initiation_method, beta=beta)
        # fit the model
        model.fit(X_train, y_train, epochs=epochs, print_cost=False)
        metric_train, _ = model.predict(X_train, y_train)  # calculate metric for training data
        metric_test, _ = model.predict(X_test, y_test)  # and for testing data
        temp_results.append([metric_train, metric_test])
    
    # Calculate averages
    avg_train = np.mean([r[0] for r in temp_results])
    avg_test = np.mean([r[1] for r in temp_results])
    
    # create a dataframe to visualize the results
    if problem_type == "classification":
        columns = [tested_param_name, 'Accuracy Train', 'Accuracy Test']
    else:
        columns = [tested_param_name, 'MSE Train', 'MSE Test']
    eval_df = pd.DataFrame([[tested_param_value, avg_train, avg_test]], columns=columns)

    return eval_df


if __name__ == "__main__":
    main()
