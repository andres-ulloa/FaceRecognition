
import numpy as np
import pandas as pd
from fully_connected_layer import classification_layer
import matplotlib.pyplot as plt 


label_position = 28 * 28
num_epochs = 5000
hidden_layer_size = 25
learning_rate = 0.0005


def find_highest_scoring_class(vector):

    sorted_list = vector.copy()
    sorted_list.sort()
    target = sorted_list[len(sorted_list) - 1]
    best_score_index = 0

    for i in range(0, len(vector)):
        if vector[i] == target:
            best_score_index = i
            break

    return best_score_index


def map_ones_to_integer(binary_vector):
    for i in range(0 , len(binary_vector)):
        if binary_vector[i] == 1:
            return i 


def classify(dataset, true_labels ,model):

    examples_counter = 0
    predictions = model.classify(dataset)
    results = []
    for num_example in range(0 , len(predictions)):

        highest_scoring_class = find_highest_scoring_class(predictions[num_example])
        label = (num_example, highest_scoring_class, map_ones_to_integer(true_labels[num_example]))
        results.append(label)

    return results


def trim_dataset(dataset, train_set_size, test_set_size):
    
    training_set = list()
    test_set = list()
    
    for i in range(0, train_set_size + test_set_size):
        if i < train_set_size:
            training_set.append(dataset[i])
        if i >= train_set_size:
            test_set.append(dataset[i])

    return training_set, test_set



def compute_confution_matrix(labels, num_classes):
    
    confution_matrix_list = list()

    for class_index  in range(0 , num_classes):

        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
            
        for label in labels:

            assigned_label = label[1]
            true_label = label[2]

            if assigned_label == true_label and assigned_label == class_index:
                true_positives += 1
                
            elif assigned_label == true_label and assigned_label != class_index:
                true_negatives += 1

            elif assigned_label !=  true_label and assigned_label != class_index:
                false_negatives += 1
                
            elif assigned_label != true_label and assigned_label == class_index:
                false_positives += 1
            
        confution_matrix = np.array([[true_positives, false_positives],[false_negatives, true_negatives]])
        confution_matrix_list.append(confution_matrix)

    return confution_matrix_list



def train(neural_net, dataset, num_epochs):

  
    print('------------------------------------------------------------------------------------------------------')
    print('----------------------------------INITIALIZING TRAINING-----------------------------------------------')
    print('------------------------------------------------------------------------------------------------------\n\n')
    print('Epochs = ', num_epochs)
    print('Alpha_rate = ', neural_net.learning_rate)
    print('Hidden layers = ', neural_net.num_hidden_layers)
    print('Input layer size = ', neural_net.input_layer_size)
    print('Hidden layer size = ', neural_net.hidden_layer_size)
    print('Output layer size = ', neural_net.output_layer_size)
    input('\nPress enter to continue...')
    print('\n\nTraining...')
    neural_net.initialize_weights()

    for i in range(0 , num_epochs):
        
        neural_net.run_shallow_activation_pass()
        neural_net.run_shallow_backpropagation()

    print('\nDone.')


def split_labels(dataset):
    
    labels = []
    new_dataset = []

    for i in range(0, len( dataset)):
        feature_vector = dataset[i]
        label = feature_vector[len(dataset[i]) - 1]
        feature_vector = np.delete(feature_vector, len(dataset[i]) - 1)
        vectorized_label = np.zeros(10)
        vectorized_label[int(label)] = 1
        labels.append(vectorized_label)
        new_dataset.append(feature_vector)
    
    return np.asarray(labels), np.asarray(new_dataset)


def ensemble_model_from_file():

    input_layer_size = 28 * 28 #there are going to be as much input neurons as there are pixels in each image
    num_classes = 10
    num_hidden_layers = 1
    global hidden_layer_size  #not considering bias units so a + 1 size in each layer should always be taken into account
    global num_epochs
    training_set_size = 900
    test_set_size = 100
    num_layers = num_hidden_layers + 1
    global learning_rate 
    neural_net = ANN(input_layer_size, num_classes, num_hidden_layers, hidden_layer_size, learning_rate, num_layers, [], []) 
    neural_net.load_weights_from_memory()
    print('Current model architecture: \n')
    print('Epochs = ', num_epochs)
    print('Alpha_rate = ', neural_net.learning_rate)
    print('Hidden layers = ', neural_net.num_hidden_layers)
    print('Input layer size = ', neural_net.input_layer_size)
    print('Hidden layer size = ', neural_net.hidden_layer_size)
    print('Output layer size = ', neural_net.output_layer_size)
    return neural_net



def demo():

    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------RUNNING DEMO--------------------------------------------------')
    print('----------------------------------------------------------------------------------------------------\n\n')
    dataset = np.genfromtxt('mnist.txt')
    neural_net = ensemble_model_from_file()
    input('\nPress enter to continue...\n')
    train_test, test_set = trim_dataset(dataset, 900, 100)    
    labels_test, dataset_test = split_labels(test_set)
    print('\nRunning retrieved model on Test set... \n')
    predictions = classify(dataset_test, labels_test, neural_net)
    print(predictions)
    results = compute_confution_matrix(predictions, 10)
    print('\nRESULTS = \n')
    for i in range(0, len(results)): print('Class ', i, '\n', results[i], '\n')


def plot_error_registry(error_registry):
    plt.plot(error_registry, linewidth = 3)
    plt.show()


def train_model():
    
    dataset = np.genfromtxt('mnist.txt')
    input_layer_size = 128
    num_classes = 10
    num_hidden_layers = 1
    global hidden_layer_size  #not considering bias units so a + 1 size in each layer should always be taken into consideration
    global num_epochs
    training_set_size = 900
    test_set_size = 100
    num_layers = num_hidden_layers + 1
    global learning_rate
    labels, dataset = split_labels(dataset)
    neural_net = ANN(input_layer_size, num_classes, num_hidden_layers, hidden_layer_size, learning_rate, num_layers, dataset, labels) 
    train(neural_net, dataset, num_epochs)
    neural_net.save_weights()
    plot_error_registry(neural_net.error_registry)


if __name__ == '__main__':
    u_input = input('Train a new architecture?(Y/N)\n')
    if u_input.lower() == 'y':
        train_model()
    else:
        demo()