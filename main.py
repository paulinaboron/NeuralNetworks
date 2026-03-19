import pandas as pd
# import matplotlib.pyplot as plt

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
    # plt.figure(figsize=(20,20))
    # sns.heatmap(corr, cmap='mako_r',annot=True)
    # plt.show()

    cor_target = abs(corr["label"])
    relevant_features = cor_target[cor_target > 0.2]
    names = relevant_features.index.tolist()

    # Drop the target variable from the results
    names.remove('label')

    n_features = len(names)
    print(names)
    print(f"Number of features: {n_features}")

    X = df[names].values
    y = df['label'].values

    # Standardization
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Example of parameter testing: Learning Rate
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    all_results = []

    for i, lr in enumerate(learning_rates, start=1):
        results = train_evaluate_model(X_train, y_train, X_test, y_test, lr, [X_train.shape[1], 16, 1], "classification", 5000)
        results.index = ['Model ' + str(i)]
        all_results.append(results)

    final_results  = pd.concat(all_results, ignore_index=False)
    print(final_results)



def solve_regression():
    print("\n=== PROBLEM REGRESJI: Exam Performance ===")
    df = pd.read_csv("StudentPerformanceFactors.csv")

    # Selection of numerical data
    features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores']
    X = StandardScaler().fit_transform(df[features].values)
    y = df['Exam_Score'].values.reshape(-1, 1)
    y = StandardScaler().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Example of parameter testing: Number of neurons in hidden layer
    neuron_configs = [4, 8, 16, 32]
    all_results = []

    for i, nc in enumerate(neuron_configs, start=1):
        results = train_evaluate_model(X_train, y_train, X_test, y_test, 0.1, [X_train.shape[1], nc, 1], "regression", 100)
        results.index = ['Model ' + str(i)]
        all_results.append(results)

    final_results  = pd.concat(all_results, ignore_index=False)
    print(final_results)


def train_evaluate_model(X_train, y_train, X_test, y_test, learning_rate, layer_dimensions, problem_type, epochs):
    """
    Keyword arguments:
    X_train -- Training data
    y_train -- Traing labels
    X_train -- test data
    y_train -- test labels
    layer_dimensions -- python array (list) containing the dimensions of each layer in our network
    problem_type -- 'classification' or 'regression'
    learning_rate --  learning rate of the network.
    Epochs -- number of iterations of the optimization loop
    returns a dataframe
    """
    # create model instance with the given hyperparameters
    model = NeuralNetwork(learning_rate=learning_rate, layer_dimensions=layer_dimensions, problem_type=problem_type)
    # fit the model
    model.fit(X_train, y_train, epochs=epochs, print_cost=True)
    accuracy, predictions = model.predict(X_test, y_test)  # calculate accuracy and predictions


    # create a dataframe to visualize the results
    columns = ['Learning_Rate', 'Layers', 'Epochs', 'Accuracy' if problem_type == "classification" else 'MSE']
    eval_df = pd.DataFrame([[learning_rate, layer_dimensions, epochs, accuracy]], columns=columns)

    return eval_df


if __name__ == "__main__":
    main()
