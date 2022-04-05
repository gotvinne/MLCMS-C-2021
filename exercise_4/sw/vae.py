
from tensorflow import keras as tfk
import tensorflow as tf
from keras import backend as kb
from keras import layers as kl
from keras import metrics as km
import matplotlib.pyplot as plt
import numpy as np

def sample_gaussian(args):
    """Sample function for a gaussion distributed variable

    Args:
        args (lst): Values describing the mean and the variance of the variable

    Returns:
        [np.ndarray]: Tensor defining the gaussian variable
    """
    z_mu, z_sigma = args
    batch = kb.shape(z_mu)[0]
    dim = kb.int_shape(z_mu)[1]
    eps = kb.random_normal(shape=(batch,dim))
    return z_mu + kb.exp(z_sigma / 2) * eps

def generate_nmist_data(train_data, test_data, num_chan, scale):
    """Generate scale and reshape nmist database for VAE

    Args:
        train_data (np.ndarray): numpy data type representing the training data
        test_data (np.ndarray): numpy data type representing the testing data
        num_chan (int): number of channels used
        scale (int): data scaling factor

    Returns:
        [np.ndarray]: Reformated training data 
        [np.ndarray]: Reformated testing data
        [lst]: Containing the dimensions used for keras layers
    """
    # Reformat data set
    x_train, x_test = train_data[0].astype('float32')*scale, test_data[0].astype('float32')*scale
    img_w, img_h  = x_train.shape[1], x_train.shape[2]

    x_train = x_train.reshape(x_train.shape[0], img_h, img_w, num_chan)
    x_test = x_test.reshape(x_test.shape[0], img_h, img_w, num_chan)

    input_shape = (img_h,img_w,num_chan)
    return x_train, x_test, input_shape

def vae_loss(x,z_decoded,z_mu,z_sigma):
    """Calculating the VAE loss consisting of a reconstruction loss and the KL divergence

    Args:
        x (np.ndarray): data 
        z_decoded (np.ndarray): data decoded from the decoder
        z_mu (): [description]
        z_sigma ([type]): [description]

    Returns:
        [np.ndarray]: [description]
    """
    x = kb.flatten(x)
    z_decoded = kb.flatten(z_decoded)
    
    # Reconstruction loss 
    recon_loss = km.binary_crossentropy(x, z_decoded)
    # KL divergence
    kl_loss = -5e-4 * kb.mean(1 + z_sigma - kb.square(z_mu) - kb.exp(z_sigma), axis=-1)
    return kb.mean(recon_loss + kl_loss)

def encoder_model(units, latent_dim, input_shape):
    """Defining the encoder NN in tensorflow

    Args:
        units (int): defining data points per sample
        latent_dim (int): defining the latent dimention
        input_shape (lst): The dimensions used for keras layers

    Returns:
        [tfk.Model]: Model representing the encoder NN
        [keras.layer.Input]: [description]
        [tuple]: Shape of the encoded layer tensor
        [keras.layer.Dense]: The mean value of a gaussian distributed variable
        [keras.layer.Dense]: The variance of a gaussian distributed variable
        [keras.layer.Lambda]: The gaussian distributed variable
    """
    encoder_input = kl.Input(shape=input_shape, name="encoder_input")
    encoder_layer = kl.Dense(units, activation=tf.nn.relu)(encoder_input)

    conv_shape = kb.int_shape(encoder_layer)
    encoder_output = kl.Flatten()(encoder_layer)

    z_mu = kl.Dense(latent_dim, name='latent_mu')(encoder_output)
    z_sigma = kl.Dense(latent_dim, name='latent_sigma')(encoder_output)
    z = kl.Lambda(sample_gaussian, output_shape=(latent_dim, ), name='z')([z_mu, z_sigma])
    encoder = tfk.Model(encoder_input, [z_mu,z_sigma,z], name="encoder")

    return encoder, encoder_input, conv_shape, z_mu, z_sigma, z

def decoder_model(units, latent_dim, conv_shape): 
    """Defining the decoder NN in tensorflow

    Args:
        units (int): defining data points per sample
        latent_dim (int): defining the latent dimention
        conv_shape (tuple): The dimensions used for keras layers

    Returns:
        [tfk.Model]: Model representing the encoder NN
    """
    decoder_input = kl.Input(shape=(latent_dim, ), name="decoder_input")
    decoder_layer = kl.Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation=tf.nn.relu)(decoder_input)
    decoder_layer = kl.Reshape((conv_shape[1],conv_shape[2],conv_shape[3]))(decoder_layer)
    decoder_output = kl.Dense(units, activation=tf.nn.relu)(decoder_layer)
    decoder_output = kl.Dense(1, activation=tf.nn.relu)(decoder_output)
    decoder = tfk.Model(decoder_input, decoder_output, name='decoder')
    return decoder

def vae_model(lr, z_mu, z_sigma, z_decoded, encoder_input):
    """Defining the vae model in tensorflow framework

    Args:
        lr (float): learning rate of the vae
        z_mu (keras.layer.Dense): The mean value of a gaussian distributed variable
        z_sigma (keras.layer.Dense): The variance of a gaussian distributed variable
        z_decoded (keras.layer.Lambda): The gaussian distributed variable decoded
        encoder_input (keras.layer.input): The input of the encoder 

    Returns:
        [tfk.Model]: Model representing the vae model
    """
    vae = tfk.Model(encoder_input, z_decoded, name='vae')

    encoder_input = kb.flatten(encoder_input)
    z_decoded = kb.flatten(z_decoded)
            
    recon_loss = km.binary_crossentropy(encoder_input, z_decoded)
    kl_loss = -5e-4 * kb.mean(1 + z_sigma - kb.square(z_mu) - kb.exp(z_sigma), axis=-1)
    total_loss= recon_loss + kl_loss

    vae.add_loss(total_loss)
    vae.add_metric(total_loss, name='loss', aggregation='mean')
    vae.compile(optimizer=tfk.optimizers.Adam(learning_rate=lr),loss=None)
    return vae

def vae(units, latent_dim, lr, input_shape):
    """Creates all the models needed to implement a variational autoencoder

    Args:
        units (int): number of data points per sample
        latent_dim (int): The dimension of the latent space
        lr (float): learning rate of the vae
        input_shape (lst): The dimensions used for keras layers

    Returns:
        [tfk.Model]: Model representing the vae model
        [tfk.Model]: Model representing the encoder NN
        [tfk.Model]: Model representing the decoder NN
        
    """
    encoder, encoder_input, conv_shape, z_mu, z_sigma, z = encoder_model(units,latent_dim,input_shape)
    decoder = decoder_model(units, latent_dim, conv_shape)
    z_decoded = decoder(z)
    vae = vae_model(lr,z_mu,z_sigma,z_decoded,encoder_input)
    return vae, encoder, decoder

def train_model(model, epochs, batch_size, train_data, test_data):
    """Training a keras model

    Args:
        model (tfk.Model): Tensorflow model to be trained
        epochs (int): Number of epochs used in training routine
        batch_size (int): Denote the size of the batch
        train_data (np.ndarray): traning data
        test_data (np.ndarray): testing data

    Returns:
        [keras.model]: The trained model
    """
    trained_model = model.fit(train_data, epochs=epochs, batch_size=batch_size, validation_data=(test_data, None))
    return trained_model 

def plot_mean_classification(x_test, y_test, encoder):
    """Plotting the classification based on the mean value

    Args:
        x_test (np.ndarray): x data test set
        y_test (np.ndarray): y data test set
        encoder (tfk.Model): Model representing the encoder NN
    """
    mu, _, _ = encoder.predict(x_test)
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(mu[:, 0], mu[:, 1], c=y_test, cmap='brg')
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.colorbar()
    plt.show()

def display_imgs(x, y=None):
    """Displays the data given in a row. 

    Args:
        x (np.array): Data array
        y (np.array, optional): data array. Defaults to None.
    """
    if not isinstance(x, (np.ndarray, np.generic)):
        x = np.array(x)
    plt.ioff()
    n = x.shape[0]
    fig, axs = plt.subplots(1, n, figsize=(n, 1))
    if y is not None:
        fig.suptitle(np.argmax(y, axis=1))
    for i in range(n):
        axs.flat[i].imshow(x[i].squeeze(), interpolation='none', cmap='gray')
        axs.flat[i].axis('off')
    plt.show()
    plt.close()
    plt.ion()

def latent_space_plot(img_w, img_h, num_chan, decoder):
    """Function plotting the latent space

    Args:
        img_w (int): sample width
        img_h (int): sample height
        num_chan (int): number of channels
        decoder (tfk.Model): Model representing the decoder NN
    """
    n = 20  # generate 15x15 digits
    figure = np.zeros((img_w * n, img_h * n, num_chan))

    grid_x = np.linspace(-5, 5, n)
    grid_y = np.linspace(-5, 5, n)[::-1]

    # Decoder for each square in the grid
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(img_w, img_h, num_chan)
            figure[i * img_w: (i + 1) * img_w,
                j * img_h: (j + 1) * img_h] = digit

    fig = plt.figure(figsize=(10, 10))
    #Reshape for visualization
    fig_shape = np.shape(figure)
    figure = figure.reshape((fig_shape[0], fig_shape[1]))

    plt.imshow(figure, cmap='gnuplot2')
    plt.show() 

