

from siamese_net import SiameseNetwork



if __name__ == '__main__':

    facenet = SiameseNetwork(0.25, (3, 96, 96))
    input_file = input('Insert a path\n')
    facenet.generate_embeddings('embeddings.csv', input_file)