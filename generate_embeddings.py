
import os
from siamese_net import SiameseNetwork
import numpy as np


def get_name(identity):

    first_name = ''
    count_under_scores = 0 
    
    for index in range(0 , len(identity)):
    
        c = identity[index]    

        if c == '_':
            count_under_scores += 1
        
        if count_under_scores == 2:
            break
        else:
            first_name += c

    return first_name


def generate_embeddings(input_dir, FRmodel):
    
        print('\n\n--------------------------------------------------------------------------')
        print('----------------------GENERATING EMBEDDINGS-------------------------------')
        print('--------------------------------------------------------------------------')
        
        embeddings = []
        classes = []
        identities = []

        image_paths = os.listdir(input_dir)

        for path in image_paths:
            
            identity = os.path.splitext(os.path.basename(path))[0]
            name = get_name(identity)
            path = input_dir + '/' + path

            print(identity) 
            embedding = FRmodel.generate_embedding(path)[0]
            print(embedding.shape)
            embeddings.append(embedding)

            classes.append(name)
            identities.append(identity)


        return embeddings, identities, classes
    

def save_to_file(list, path):

    with open(path, 'w') as f:
        for item in list:
            f.write("%s\n" % item)

        

if __name__ == '__main__':

    facenet = SiameseNetwork(0.25, input_shape = (3, 96, 96))
    embeddings, identities, classes = generate_embeddings('pre_processed_test_set/', facenet)
    print('\nSaving...')
    np.savetxt('test_embeddings.csv', embeddings, delimiter = ',')
    save_to_file(identities, 'test_identities.txt')
    save_to_file(classes, 'test_classes.txt')
    print('\nDone.')