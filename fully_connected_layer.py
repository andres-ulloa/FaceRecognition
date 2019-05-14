import numpy as np
import math


def sigmoid(x):
    return 1.0/(1+ np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class classification_layer:
    
    def __init__(self, input_layer_size, output_layer_size, num_hidden_layers, hidden_layer_size, learning_rate, num_layers, input_, labels):
        
        self.labels = np.vstack(labels)
        self.input = np.vstack(input_) 
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.num_hidden_layers = num_hidden_layers             
        self.hidden_layer_size = hidden_layer_size 
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.error_registry = list()


    def initialize_weights(self):

        self.weights_layer_1 = np.random.uniform(low = -0.5, high = 0.5, size = (self.input.shape[1], self.hidden_layer_size))
        self.weights_layer_2 = np.random.uniform(low = -0.5, high = 0.5, size = (self.hidden_layer_size,self.output_layer_size))  
        self.bias_unit = 1


    def load_weights_from_memory(self):
       print('\nRetrieving weights from: params_layer_1_csv, params_layer_2.csv, bias_unit.csv')
       print('...')
       print('...\n')
       self.weights_layer_1 =  np.loadtxt('fcl_weights_1.csv' , delimiter = ',')
       self.weights_layer_2 =  np.loadtxt('fcl_weights_2.csv' , delimiter = ',')
       self.bias_unit =  np.loadtxt('bias_unit.csv' , delimiter = ',')
       print(self.weights_layer_1.shape)
       print(self.weights_layer_2.shape)
       print(self.bias_unit)
       print('\n\nDone.\n\n')
    

    def save_weights(self):
        print('\nSaving model parameters...')
        bias_unit = list()
        bias_unit.append(self.bias_unit)
        np.savetxt('params_layer_1.csv', self.weights_layer_1, delimiter = ',')
        np.savetxt('params_layer_2.csv', self.weights_layer_2, delimiter = ',')
        np.savetxt('bias_unit.csv', bias_unit, delimiter = ',')
        print('\nDone.')


    def classify(self, dataset):
        self.input = dataset
        self.run_shallow_activation_pass()
        return self.output



    def run_shallow_activation_pass(self):
        
        self.activation_layer_1 = sigmoid(np.dot(self.input, self.weights_layer_1))
       
        self.output = sigmoid(np.dot(self.activation_layer_1, self.weights_layer_2) + self.bias_unit)
      
    

    def run_shallow_backpropagation(self):
        error = self.labels - self.output
        global_error_derivative = (2 * (error) * sigmoid_derivative(self.output))
        
        global_cost = np.sum(error)
        self.error_registry.append(global_cost)

        layer_2_gradient = np.dot(self.activation_layer_1.T, (2*(self.labels - self.output) * sigmoid_derivative(self.output)))
        layer_1_gradient = np.dot(self.input.T, (np.dot(2*(self.labels - self.output) * sigmoid_derivative(self.output), self.weights_layer_2.T) * sigmoid_derivative(self.activation_layer_1)))

        b_gradient = np.sum(global_error_derivative)

        self.weights_layer_1 += layer_1_gradient * self.learning_rate
        self.bias_unit += b_gradient * self.learning_rate
        self.weights_layer_2 += layer_2_gradient * self.learning_rate





        
        
        