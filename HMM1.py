import sys

def get_matrix(input):
    data = input.strip().split()
    rows, cols = int(data[0]), int(data[1])
    vals = list(map(float, data[2:]))
    matrix = []
    for i in range(rows):
        matrix.append(vals[i * cols:(i + 1) * cols])
    return rows, cols, matrix

#estimate the probability of an observation sequence
def forward_algorithm(transition_matrix, emission_matrix, initial_state_probability_distribution, observation_sequence):
    
    initial_distribution = initial_state_probability_distribution[0]
   
    num_states = len(transition_matrix) 
    num_observ = len(observation_sequence)

    alpha = []
    for i in range(num_observ):  
        row = [0] * num_states  
        alpha.append(row) 
    
    for i in range(num_states):
        alpha[0][i] = emission_matrix[i][observation_sequence[0]] * initial_distribution[i]


    for t in range(1, num_observ):
        for i in range(num_states):
            sum_alpha = 0
            for j in range(num_states):
                sum_alpha += alpha[t - 1][j] * transition_matrix[j][i]
            
            #multiply the sum of alpha values with the emission probability
            #updates the probability of being in state i at time t
            alpha[t][i] = emission_matrix[i][observation_sequence[t]] * sum_alpha

    #sum the final alpha values for all states to get the total probability of the observation sequence
    final_prob = sum(alpha[num_observ - 1][i] for i in range(num_states))
    return final_prob
    

inputs = sys.stdin.read().strip().split("\n")

transition_rows, transition_cols, transition_matrix = get_matrix(inputs[0])
# print(transition_matrix)
emission_rows, emission_cols, emission_matrix = get_matrix(inputs[1])
# print(emission_matrix)
initial_rows, initial_cols, initial_distribution = get_matrix(inputs[2])
# print(initial_distribution)
observation_sequence = list(map(int, inputs[3].split()[1:]))
# print(observation_sequence)

# transition_matrix = [[0.0, 0.8, 0.1, 0.1], [0.1, 0.0, 0.8, 0.1], [0.1, 0.1, 0.0, 0.8], [0.8, 0.1, 0.1, 0.0]]
# emission_matrix = [[0.9, 0.1, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.9, 0.1], [0.1, 0.0, 0.0, 0.9]]
# initial_distribution = [[1.0, 0.0, 0.0, 0.0]]
# observation_sequence = [0, 1, 2, 3, 0, 1, 2, 3]

prob = forward_algorithm(transition_matrix, emission_matrix, initial_distribution, observation_sequence)

print(prob)

