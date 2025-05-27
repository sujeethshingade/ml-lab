import pandas as pd

def find_s_algorithm(file_path):
    data = pd.read_csv(file_path)

    print("Training data:\n", data)

    data = data.drop(columns=['day'])
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    hypothesis = None

    for i in range(len(data)):
        if y[i] == 'Yes':
            if hypothesis is None:
                hypothesis = list(X.iloc[i])
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j] != X.iat[i, j]:
                        hypothesis[j] = '?'

    return hypothesis

# Run the algorithm
file_path = 'play_tennis.csv'
final_hypothesis = find_s_algorithm(file_path)
print("\nFinal hypothesis:", final_hypothesis)
