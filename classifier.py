

import numpy as np
import math
from sklearn.svm import SVC


def train_and_save_classifier(emb_array, label_array, class_names, classifier_filename_exp):

    print('Training Classifier')
    model = SVC(kernel = 'linear', probability = True, verbose = False)
    model.fit(emb_array, label_array)
    with open(classifier_filename_exp, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)
    print('Saved classifier model to file "%s"' % classifier_filename_exp)

    return model

def run_cross_validation(model, folds):
    pass


def test_classifier(model, test_set):
    pass

def load_dataset(path_data, labels, classes):
    path_data, labels, classes = None

    return path_data, labels, classes


def load_model_parameters(file_path):
    pass


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



def generate_new_model():

    embeddings, classes, labels = load_dataset()
    svm = train_and_save_classifier(embeddings, labels, class_names, 'svm_weights.csv')
    confution_matrix =  compute_confution_matrix(labels)
    print(confution_matrix)


def run_demo():
    
    print('------------------------------------------------------------------------------------------------------')
    print('--------------------------------------RUNNING DEMO-----------------------------------------------------')
    print('----------------------------------------------------------------------------------------------------\n\n')
    


if __name__ == '__main__':

    new_model = input('Train a new model? (y/n)')

    if new_model == 'y':
        generate_new_model()
    else:
        run_demo()
