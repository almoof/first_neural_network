import numpy as np
rng = np.random.default_rng()

learning_rate = 0.5
number_of_layers = 2

layer_sizes = [2, 2, 1]

Weights = []
Biases = []
Training_answers = [np.array([[1]]), np.array([[0]]), np.array([[1]]), np.array([[0]])] # Cat = 1, Dog = 0

for l in range(0, number_of_layers):
   Weights.append(rng.random((layer_sizes[l+1], layer_sizes[l])))
   Biases.append(np.zeros((layer_sizes[l+1], 1)))

# print(Weights)
# print(Biases)

Inputs = [np.array([[1], [3]]), np.array([[3], [0]]), np.array([[0], [5]]), np.array([[0], [0]])]

# print(Inputs)

Activations = [[] for _ in range(len(Inputs))]
Weighted_input = [[] for _ in range(len(Inputs))]
Loss_functions = []
Cost = 0
Deltas = [[] for _ in range(len(Inputs))] # задом наперед, спереди дельта за последний слой и т.д.
Costs_to_biases = [[] for _ in range(len(Inputs))]
Costs_to_weights = [[] for _ in range(len(Inputs))]
Deltas_fixed = []

def sigmoid(x):
   return 1/(1+np.exp(-x))

def loss(x):
  return (Training_answers[x] - Activations[x][2]) ** 2

def delta_L(x):
  return (Activations[x][2]-Training_answers[x])*sigmoid(Weighted_input[x][1])*(1 - sigmoid(Weighted_input[x][1]))

def delta_l(x, l):
  return (np.transpose(Weights[l+2]) @ Deltas[x][l+1]) * sigmoid(Weighted_input[x][l+1])*(1 - sigmoid(Weighted_input[x][l+1])) # weight_L -1+2=1 в массиве второй, а delta_L в массиве первый так что -1+1=0

def partial_derivative_b(x, l):
  return Deltas_fixed[x][l]

def partial_derivative_w(x, l):
  return Deltas_fixed[x][l] @ np.transpose(Activations[x][l])

def forward(Input):
  for i in range(len(Inputs)):
    Activations[i].append(Input[i])

  for x in range(0, len(Inputs)): # 3 training examples
    for i in range(0, number_of_layers):
      # print(Weights[i].shape)
      Z = Weights[i] @ Activations[x][i] + Biases[i]
      Weighted_input[x].append(Z)
      A = sigmoid(Z)
      Activations[x].append(A)
  # print("y1_hat = ", Activations[0][2])
  # print("y2_hat = ", Activations[1][2])
  # print("W = ", Weights)
  # print("b = ", Biases)

def backpropagation():
  global Cost
  global Deltas_fixed
  for x in range(0, len(Inputs)):
    Loss_functions.append(loss(x))
  Cost = sum(Loss_functions)/len(Inputs)
  # print(Cost)
  # print(Loss_functions)
  for x in range(0, len(Inputs)):
    Deltas[x].append(delta_L(x))
  
  for x in range(0, len(Inputs)):
    for l in range(-1, 0):
      Deltas[x].append(delta_l(x, l))
  # print(Deltas)
  Deltas_fixed = [row[::-1] for row in Deltas] # уже не задом наперед
  # print(Deltas_fixed)
  for x in range(0, len(Inputs)):
    for l in range(0, 2):
      Costs_to_biases[x].append(partial_derivative_b(x, l))
  #print(Costs_to_biases)

  for x in range(0, len(Inputs)):
    for l in range(0, 2):
      Costs_to_weights[x].append(partial_derivative_w(x, l))
  #print(Costs_to_weights)

  for x in range(0, len(Inputs)):
    for l in range(0, 2):
      Weights[l] = Weights[l] - learning_rate * Costs_to_weights[x][l]
  # print("W = ", Weights)

  for x in range(0, len(Inputs)):
    for l in range(0, 2):
      Biases[l] = Biases[l] - learning_rate * Costs_to_biases[x][l]
  # print("b = ", Biases)


forward(Inputs)
print("W = ", Weights)
print("b = ", Biases)
print("y1_hat = ", Activations[0][2])
print("y2_hat = ", Activations[1][2])
print("y3_hat = ", Activations[2][2])
print("y4_hat = ", Activations[3][2])

for i in range(0, 10000):
  
  Activations = [[] for _ in range(len(Inputs))]
  Weighted_input = [[] for _ in range(len(Inputs))]
  Loss_functions = []
  Cost = 0
  Deltas = [[] for _ in range(len(Inputs))] # задом наперед, спереди дельта за последний слой и т.д.
  Costs_to_biases = [[] for _ in range(len(Inputs))]
  Costs_to_weights = [[] for _ in range(len(Inputs))]
  Deltas_fixed = []


  forward(Inputs)
  backpropagation()
  if i%2000 == 0:
    print(Cost)
  


Activations = [[] for _ in range(len(Inputs))]
Weighted_input = [[] for _ in range(len(Inputs))]
forward(Inputs)

print("W = ", Weights)
print("b = ", Biases)
print("y1_hat = ", Activations[0][2])
print("y2_hat = ", Activations[1][2])
print("y3_hat = ", Activations[2][2])
print("y4_hat = ", Activations[3][2])