import matplotlib.pyplot as plt
import tensorflow as tf


print(tf.__version__) # 2.5.0 in case of Colab

(train_X, train_Y), (test_X, test_Y) =  tf.keras.datasets.mnist.load_data()
train_X = train_X/255.0
test_X = test_X/255.0

train_X = train_X.reshape(-1,28,28,1)
test_X = test_X.reshape(-1,28,28,1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=(2, 2), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=(2, 2), activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dense(7 * 7 * 64, activation='elu'),
    tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
    tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=(2, 2), padding="same", activation="elu"),
    tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=2, strides=(2, 2), padding="same", activation="sigmoid"),

])

latent_vector_model = tf.keras.Model(inputs = model.input, outputs = model.layers[3].output)
latent_vector = latent_vector_model.predict(train_X)

print(latent_vector.shape) # (60000, 64)
print(latent_vector[0])
# [ 0.03989753  0.07165951  0.04068347 -0.04980791  0.0237006   0.04206682
#   0.12236142  0.03561687 -0.02874464 -0.03207129 -0.05090731 -0.00033051
#   0.00059745  0.02293842  0.0194317  -0.0040431  -0.002298    0.03510047
#  -0.05498803  0.05566446  0.01128561 -0.00385684  0.05775554  0.05043943
#  -0.0950135  -0.03467274 -0.00893998 -0.04763365  0.00160492 -0.10830277
#   0.10279456  0.07270341  0.03029765 -0.09919685 -0.03164804  0.00397745
#  -0.03356767 -0.09055948 -0.02668971  0.04829129 -0.01274872  0.07275461
#   0.03396153  0.02727858 -0.05546844 -0.08266592  0.04146755  0.03477928
#  -0.05506766 -0.05510765  0.04873044 -0.01005602 -0.018543    0.12394038
#  -0.06747854 -0.01007408 -0.04212677  0.05144825  0.01496188  0.00751211
#   0.02602987 -0.06681389 -0.01051581  0.02298225]


###########################################################################
# KMeans
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 10, n_init = 100, random_state = 42)
kmeans.fit(latent_vector)

print(kmeans.labels_) # [3 2 9 ... 3 8 6]
print(kmeans.cluster_centers_.shape) # (10, 64)
print(kmeans.cluster_centers_[0])

# [-0.00583038  0.02354546  0.00330501 -0.06204341 -0.00909809 -0.0033203
#   0.03917303  0.04829047 -0.02968106 -0.02459479 -0.00802828  0.03655537
#   0.02265877  0.01717704  0.04120917  0.00330051  0.01515519  0.02694755
#  -0.01563619  0.02530236  0.01199479  0.01790801  0.04124565  0.00880383
#  -0.01078923 -0.02190818  0.03007521  0.01126036 -0.01439181 -0.0606074
#   0.0169812   0.0352301  -0.0290094  -0.01378469 -0.01372908 -0.02855882
#  -0.05447526  0.04068633  0.00330245  0.03782793  0.02107317  0.04585927
#  -0.02463678 -0.0429853  -0.00586792 -0.02949569 -0.01520366  0.00901652
#  -0.01923866 -0.02747743  0.01601529  0.0002512  -0.03692435  0.01675072
#  -0.0116922  -0.0578577   0.01183607  0.0085212   0.00658136  0.00148519
#   0.01085111 -0.03688751 -0.04689862 -0.02039468]


plt.figure(figsize=(12, 12))

for i in range(10):
    images = train_X[kmeans.labels_ == i]
    for c in range(10):
        plt.subplot(10, 10, i * 10 + c + 1)
        plt.imshow(images[c].reshape(28, 28), cmap='gray')
        plt.axis('off')

plt.show()


# t-SNE
from sklearn.manifold import TSNE
import numpy as np

tsne = TSNE(n_components=2, learning_rate = 100, perplexity = 15, random_state = 0)
tsne_vector = tsne.fit_transform(latent_vector[:5000])


cmap = plt.get_cmap('rainbow',10)
fig = plt.scatter(tsne_vector[:,0], tsne_vector[:,1], marker = '.', c = train_Y[:5000], cmap = cmap)
cb = plt.colorbar(fig, ticks = range(10))
n_clusters = 10
tick_locs = (np.arange(n_clusters) + 0.5)* (n_clusters - 1) / n_clusters



cb.set_ticks(tick_locs)
cb.set_ticklabels(range(10))
plt.show()

perplexities = [5, 10, 15, 25, 50, 100]
plt.figure(figsize=(8, 12))

for c in range(6):
    tsne = TSNE(n_components=2, learning_rate=100, perplexity=perplexities[c], random_state=0)
    tsne_vector = tsne.fit_transform(latent_vector[:5000])

    plt.subplot(3, 2, c + 1)
    plt.scatter(tsne_vector[:, 0], tsne_vector[:, 1], marker='.', c=train_Y[:5000], cmap='rainbow')
    plt.title('perplexity : {0}'.format(perplexities[c]))

plt.show()




