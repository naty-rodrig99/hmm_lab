import sys

def get_matrix(input):
    data = input.strip().split()
    rows, cols = int(data[0]), int(data[1])
    vals = list(map(float, data[2:]))
    matrix = []
    for i in range(rows):
        matrix.append(vals[i * cols:(i + 1) * cols])
    return rows, cols, matrix

#computes the most probable sequence of hidden states
def viterbi(transition_matrix, emission_matrix, initial_state_probability_distribution, emission_sequence):
    
    initial_distribution = initial_state_probability_distribution[0]
   
    num_states = len(transition_matrix)
    num_time_steps = len(emission_sequence)
    
    delta = []
    for i in range(num_time_steps):  
        row = [0] * num_states
        delta.append(row) 

    backtrack = []
    for i in range(num_time_steps):
        row = [0] * num_states
        backtrack.append(row)

    for state in range(num_states):
        delta[0][state] = emission_matrix[state][emission_sequence[0]] * initial_distribution[state]
        backtrack[0][state] = 0 

    for t in range(1, num_time_steps):
        for state in range(num_states):
            max_prob = 0
            best_prev_state = 0
            observ = emission_sequence[t]
            emission_prob = emission_matrix[state][observ]
            
            for prev_state in range(num_states):
                prob = delta[t-1][prev_state] * transition_matrix[prev_state][state]
                if prob > max_prob:
                    max_prob = prob
                    best_prev_state = prev_state
            
            #stores max probability of being in state i at time t, given observations
            delta[t][state] = max_prob * emission_prob

            #stores the state that led the optimal path at time t for state i
            backtrack[t][state] = best_prev_state
        
        max_prob = max(delta[-1])
        best_last_state = delta[-1].index(max_prob)

    best_path = [0] * num_time_steps
    best_path[-1] = best_last_state
    for t in range(num_time_steps-2,-1,-1):
        best_path[t] = backtrack[t+1][best_path[t+1]]
        
    return best_path
    

inputs = sys.stdin.read().strip().split("\n")

transition_rows, transition_cols, transition_matrix = get_matrix(inputs[0])
# print(transition_matrix)
emission_rows, emission_cols, emission_matrix = get_matrix(inputs[1])
# print(emission_matrix)
initial_rows, initial_cols, initial_distribution = get_matrix(inputs[2])
# print(initial_distribution)
emission_sequence = list(map(int, inputs[3].split()[1:]))
# print(emission_sequence)

# transition_matrix = transition_matrix = [[0.0, 0.8, 0.1, 0.1], [0.1, 0.0, 0.8, 0.1], [0.1, 0.1, 0.0, 0.8], [0.8, 0.1, 0.1, 0.0]]
# emission_matrix = [[0.9, 0.1, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.9, 0.1], [0.1, 0.0, 0.0, 0.9]]
# initial_distribution = [[1.0, 0.0, 0.0, 0.0]]
# emission_sequence = [1,1,2,2]

best_path = viterbi(transition_matrix, emission_matrix, initial_distribution, emission_sequence)

output = " ".join(map(str, best_path))

print(output)

