import sys

def get_matrix(input):
    data = input.strip().split()
    rows, cols = int(data[0]), int(data[1])
    vals = list(map(float, data[2:]))
    matrix = []
    for i in range(rows):
        matrix.append(vals[i * cols:(i + 1) * cols])
    return rows, cols, matrix

#calculates the probability of an observation sequence given the model parameters
def forward_algorithm(transition_matrix, emission_matrix, initial_distribution, observation_sequence):
    num_states = len(transition_matrix)
    num_observ = len(observation_sequence)
    c = 0
    alpha = []
    for i in range(num_observ):
        alpha.append([0] * num_states)
    
    c = [0] * num_observ

    #initialize - compute the alpha[0][i]
    for i in range(num_states):
        alpha[0][i] = initial_distribution[0][i] * emission_matrix[i][observation_sequence[0]]
        c[0] += alpha[0][i]
    
    #scale the alpha[0][i]
    c[0] = 1 / c[0]
    for i in range(num_states):
        alpha[0][i] *= c[0]

    #compute alpha[t][i]
    for t in range(1, num_observ):
        c[t] = 0
        for i in range(num_states):
            alpha[t][i] = 0
            for j in range(num_states):
                alpha[t][i] += alpha[t - 1][j] * transition_matrix[j][i]
            alpha[t][i] *= emission_matrix[i][observation_sequence[t]]
            c[t] += alpha[t][i]
        #scale alpha[t][i]
        c[t] = 1 / c[t]
        for i in range(num_states):
            alpha[t][i] *= c[t]

    return alpha, c

def backward_algorithm(transition_matrix, emission_matrix, observation_sequence, c):
    num_states = len(transition_matrix)
    num_observations = len(observation_sequence)

    beta = []
    for i in range(num_observations):
        beta.append([0] * num_states)

    #initialize
    for i in range(num_states):
        beta[num_observations - 1][i] = c[num_observations - 1]

    for t in range(num_observations - 2, -1, -1):
        for i in range(num_states):
            beta[t][i] = 0
            for j in range(num_states):
                beta[t][i] += (transition_matrix[i][j] * emission_matrix[j][observation_sequence[t + 1]] * beta[t + 1][j])
            beta[t][i] *= c[t] #scale with same scale factor as alpha
    return beta

#maximize the likehood of the observed sequence - no direct knowledge of hidden states
def baum_welch(transition_matrix, emission_matrix, initial_distribution, observation_sequence, max_iters=100):
    num_states = len(transition_matrix)
    num_symbols = len(emission_matrix[0])
    num_observations = len(observation_sequence)

    for k in range(max_iters):
        alpha, c = forward_algorithm(transition_matrix, emission_matrix, initial_distribution, observation_sequence)
        beta = backward_algorithm(transition_matrix, emission_matrix, observation_sequence, c)

        #initialize gamma and xi
        gamma = []
        for i in range(num_observations):
            gamma.append([0] * num_states)
        xi = []
        for i in range(num_observations - 1):
            row = []
            for j in range(num_states):
                row.append([0] * num_states)
            xi.append(row)

        #gamma - prob of being in a given state at specific time
        #xi - prob of transitioning from one state to another at a specific time step
        for t in range(num_observations - 1):
            for i in range(num_states):
                gamma[t][i] = 0
                for j in range(num_states):
                    gamma[t][i] += alpha[t][i] * transition_matrix[i][j] * emission_matrix[j][observation_sequence[t + 1]] * beta[t + 1][j]
                #prob of beiing in state i at time t and transitioning to state j at time t+1
                for j in range(num_states):
                    xi[t][i][j] = (alpha[t][i] * transition_matrix[i][j] * emission_matrix[j][observation_sequence[t + 1]] * beta[t + 1][j])
   
        for i in range(num_states):
            gamma[num_observations - 1][i] = alpha[num_observations - 1][i]

        #re-estimate xi
        for i in range(num_states):
            initial_distribution[0][i] = gamma[0][i]

        #re-estimate A
        for i in range(num_states):
            for j in range(num_states):
                denom = 0
                for t in range(num_observations - 1):
                    denom += gamma[t][i]
                numer = 0
                for t in range(num_observations - 1):
                    numer += xi[t][i][j]
                transition_matrix[i][j] = numer / denom

        #re-estimate B
        for i in range(num_states):
            for j in range(num_symbols):
                denom = 0
                for t in range(num_observations):
                    denom += gamma[t][i]
                numer = 0
                for t in range(num_observations):
                    if observation_sequence[t] == j:
                        numer += gamma[t][i]
                emission_matrix[i][j] = numer / denom

    return transition_matrix, emission_matrix

inputs = sys.stdin.read().strip().split("\n")


transition_rows, transition_cols, transition_matrix = get_matrix(inputs[0])
#print(transition_matrix)
emission_rows, emission_cols, emission_matrix = get_matrix(inputs[1])
#print(emission_matrix)
initial_rows, initial_cols, initial_distribution = get_matrix(inputs[2])
#print(initial_distribution)
observation_sequence = list(map(int, inputs[3].split()[1:]))
#print(observation_sequence)

transition_matrix_output, emission_matrix_output = baum_welch(transition_matrix, emission_matrix, initial_distribution, observation_sequence)

#print("OUT1: ",transition_matrix_output)
#print("OUT2: ",emission_matrix_output)

output1 = []
output1.append(f"{len(transition_matrix_output)} {len(transition_matrix_output[0])}")
for row in transition_matrix_output:
    for value in row:
        output1.append(f"{value:.7f}")
output1 = " ".join(output1)

output2 = []
output2.append(f"{len(emission_matrix_output)} {len(emission_matrix_output[0])}")
for row in emission_matrix_output:
    for value in row:
        output2.append(f"{value:.7f}")
output2 = " ".join(output2)

final_output = output1 + "\n" + output2

print(final_output)