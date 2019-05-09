import numpy as np
import math
from sklearn.svm import SVC

def train_and_save_classifier(emb_array, label_array, class_names, classifier_filename_exp):
    logger.info('Training Classifier')
    model = SVC(kernel='linear', probability = True, verbose = False)
    model.fit(emb_array, label_array)
    with open(classifier_filename_exp, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)
    logging.info('Saved classifier model to file "%s"' % classifier_filename_exp)


def test_classifier(model, test_set):
    pass

def load_embeddings(path_file):
    pass


def load_model_parameters(file_path):
    pass

    
def compute_confution_matrix(labels):
    pass

def main():

    embeddings = load_embeddings()
    svm = train_and_save_classifier(embeddings, labels, class_names, 'training_results.csv')
    confution_matrix =  compute_confution_matrix(labels)
    print('\n\n--------------------------------------------------------------------------------')
    print('-------------------------GENERATING ROC METRICS----------------------------------')
    print('--------------------------------------------------------------------------------\n\n')
    print(confution_matrix)



if __name__ == '__main__':
    main()
