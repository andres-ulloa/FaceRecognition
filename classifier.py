import numpy as np
import math
from sklearn.svm import SVC

def build_svm_model():
    pass

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
