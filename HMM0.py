import sys

def get_matrix(input):
    data = input.strip().split()
    rows, cols = int(data[0]), int(data[1])
    vals = list(map(float, data[2:]))
    matrix = []
    for i in range(rows):
        matrix.append(vals[i * cols:(i + 1) * cols])
    return rows, cols, matrix

def next_observation_distribution(transition_matrix, emission_matrix, initial_state_probability_distribution):
    
    initial_state_probability_distribution = initial_state_probability_distribution[0]
    new_state = [0] * len(transition_matrix)
    
    for i in range(len(transition_matrix)):  
        tmp = 0 
        for j in range(len(new_state)): 
            tmp += transition_matrix[j][i] * initial_state_probability_distribution[j]
        new_state[i] = tmp
    
    next_observation = [0] * len(emission_matrix[0])

    
    for i in range(len(emission_matrix[0])):  
        tmp = 0 
        for j in range(len(emission_matrix)):
            tmp += emission_matrix[j][i] * new_state[j]
        next_observation[i] = tmp
    # print(next_observation)
    return next_observation
    

inputs = sys.stdin.read().strip().split("\n")

transition_rows, transition_cols, transition_matrix = get_matrix(inputs[0])
# print(transition_matrix)
emission_rows, emission_cols, emission_matrix = get_matrix(inputs[1])
# print(emission_matrix)
initial_rows, initial_cols, initial_distribution = get_matrix(inputs[2])
# print(initial_distribution)

# transition_matrix = [[0.2, 0.5, 0.3, 0.0], [0.1, 0.4, 0.4, 0.1], [0.2, 0.0, 0.4, 0.4], [0.2, 0.3, 0.0, 0.5]]
# emission_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.2, 0.6, 0.2]]
# initial_distribution = [[0.0, 0.0, 0.0, 1.0]]

next_observation = next_observation_distribution(transition_matrix, emission_matrix, initial_distribution)
# print(next_observation)

formatted_values = " ".join(str(value) for value in next_observation)

num_rows = 1
num_cols = len(next_observation)
output = f"{num_rows} {num_cols} {formatted_values}"

print(output)

