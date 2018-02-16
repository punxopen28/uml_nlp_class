from pyfasttext import FastText
from sklearn.cluster import KMeans
import numpy as np 

def main():
    model = FastText('model_text8.bin')

    target_words = ['granada', 'python', 'harmony', 'mafia', 'yoga', 'goth', 
                    'cyberpunk', 'nasa', 'japan', 'boolean', 'foodball',
                    'algorithm', 'china', 'usa', 'internet', 'harvard',
                    'earth', 'horse', 'angel', 'rock']   
    for t_word in target_words:
        # get embedding
        target_word_embedding = model.get_numpy_vector(t_word)
        print('Target word:', t_word)
        #print('Embedding shape:', target_word_embedding.shape)
        #print('Embedding:', target_word_embedding[0:10], '...')
    
        # find closest words
        closest_words = model.nearest_neighbors(t_word, k=15)
        # init array
        nn_word_embedding= np.zeros(shape=(15,128))
        i = 0
        for word, similarity in closest_words:
            # get each word embedding
            nn_word_embedding[i] = model.get_numpy_vector(word)
            #print('Word:', word, 'Vec:', nn_word_embedding[i])
            i = i + 1
        # kmeans
        #print(nn_word_embedding.shape)
        #print(closest_words) 
        cluster_model = KMeans(n_clusters=3, init='k-means++')
        prediction = cluster_model.fit_predict(nn_word_embedding)
        print(prediction)
        j = 0
        for word in closest_words:
            print('Word:', word[0], '- Cluster #%d' % (prediction[j] + 1))
            j = j + 1
    

if __name__ == '__main__':
    main()
