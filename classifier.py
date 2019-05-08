import numpy as np
import math
from sklearn.svm import SVC

def _train_and_save_classifier(emb_array, label_array, class_names, classifier_filename_exp):
    logger.info('Training Classifier')
    model = SVC(kernel='linear', probability=True, verbose=False)
    model.fit(emb_array, label_array)
    with open(classifier_filename_exp, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)
    logging.info('Saved classifier model to file "%s"' % classifier_filename_exp)
    

def classify():
    pass

def load_embeddings():
    pass

def compute_confution_matrix(labels):
    pass

def main():

    embeddings = load_embeddings()
    svm = build_svm_model()
    labels = classify()
    confution_matrix =  compute_confution_matrix(labels)
    print('\n\n--------------------------------------------------------------------------------')
    print('-------------------------GENERATING ROC METRICS----------------------------------')
    print('--------------------------------------------------------------------------------\n\n')
    print(confution_matrix)



if __name__ == '__main__':
    main()
