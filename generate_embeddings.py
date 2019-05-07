

from siamese_net import SiameseNetwork



if __name__ == '__main__':

    facenet = SiameseNetwork(0.25, (3, 96, 96))
    facenet.generate_embeddings()