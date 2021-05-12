# Implements a variational auto-encoder for the MNIST problem.
#
# Authors: Daniel Hernandez Lobato
#          Luis Antonio Ortega Andrés
#          Juan Ignacio Álvarez Trejos

import autograd.numpy as np
import autograd.numpy.random as npr

from autograd import grad
from data import load_mnist
from data import save_images as s_images
from autograd.misc import (
    flatten,
)  # This is used to flatten the params (transforms a list into a numpy array)


def save_images(images, file_name=None):
    """Calls s_images with fixed parameters.

    Arguments:
     - images: Array with one row per image
     - file_name: png file in which to save the image
    """
    return s_images(images, file_name, vmin=0.0, vmax=1.0)


def sigmoid(x):
    """Sigmoid activation function to estimate probabilities"""
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    """Relu activation function for non-linearity."""
    return np.maximum(0, x)


def init_net_params(layer_sizes, scale=1e-2):
    """This function intializes the parameters of a deep neural network
    using Gaussian random parameters.

    Arguments:
     - layer_sizes: array-like with size of each layer of the NN
     - scale: float. Variance of the Gaussian generator.

    Returns:
     - array of (weights, biases) for each layer.
    """
    return [
        # (weight matrix, bias vector)
        (scale * npr.randn(m, n), scale * npr.randn(n))
        for m, n in zip(layer_sizes[:-1], layer_sizes[1:])
    ]


def batch_normalize(batch):
    """Applies batch normalization to the given batch as
    ```
      (batch - mu)/(std + 0.001)
    ```
    over the first axis.

    Arguments:
     - batch: Array like of shape [batch_size,...]

    Returns:
     - Normalized batch
    """
    # Compute mean and standard deviation.
    batch_mean = np.mean(batch, axis=0, keepdims=True)
    batch_std = np.std(batch, axis=0, keepdims=True)
    # Adding 1 to avoid dividing by zero in any situation.
    return (batch - batch_mean) / (batch_std + 1)


def neural_net_predict(params, inputs, normalize=True):
    """Computes the output of a deep neural network with params a
    list with pairs of weights and biases

    Arguments:
     - Params: list of (weights, bias) tuples.
     - inputs: (N x D) matrix.
     - normalize: boolean. Whether to apply batch normalization or not.

    Returns:
     - outputs: np.darray of shape (params[-1].shape)

    Applies batch normalization to every layer but the last.
    """

    for W, b in params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        if normalize:
            outputs = batch_normalize(outputs)
        inputs = relu(outputs)  # nonlinear transformation

    # Last layer is linear
    outW, outb = params[-1]
    outputs = np.dot(inputs, outW) + outb

    return outputs


def sample_latent_variables_from_posterior(encoder_output):
    """Generates a sample from q(z|x) per each batch datapoint
    using the reparameterization trick:

    ```
       \mathcal{N}(\mu, \sigma^2) = \mu + \mathcal{N}(0, I) * \sigma
    ```

    Arguments:
     - encoder_output: array-like of dimension (N , 2D). Encodes the
        mean and std values in the first and second half of the last
        axis respectively.

    Returns:
     - z: Random sample of diagonal gaussian distirbution of mean
          and variances from input.
    """

    D = np.shape(encoder_output)[-1] // 2
    # The first half corresponds to the mean and the second to the log std
    mean, log_std = encoder_output[:, :D], encoder_output[:, D:]

    # The output of this function is a matrix of size the batch x the number of
    # latent dimensions
    z = npr.randn(*mean.shape)
    z = mean + z * np.exp(log_std)

    return z


def bernoulli_log_prob(targets, logits):
    """
    Computes the log probability of the targets given the generator output
    specified in logits:

    ```
        \log P(x \mid z) = \sum(\log(x * probs + (1-x)*(1-probs)))
    ```
    with x in targets and logits = log(probs).

    Arguments:
     - logits. Real values. Unnormalized log probabilities
     - targets. Values in [0, 1].

    Returns
     - log_prob: Array of size batch_size. Contains the log probability
                 of each sample in the batch.

    """
    # Compute probabilities
    probs = sigmoid(logits)
    # Compute log probs
    log_prob = np.log(targets * probs + (1 - targets) * (1 - probs))

    # Sum the probabilities across the dimensions of each image in the batch.
    return np.sum(log_prob, axis=-1)


def compute_KL(q_means_and_log_stds):
    """Compute the KL divergence between q(z|x) and the prior N(0, 1).
    The KL divervence is the sum of KL divergence of
    the marginals if q and p factorize:
    ```
        KL = \frac{1}{2} \sigma^2 + \mu^2 - 1 - \log(\sigma^2).
    ```

    Arguments:
     - q_means_and_log_stds: numpy np.darray of shape (N, 2D).
        Learned mean and log std of each sample in batch size.
        The first (:, D) values retain the means and the (:, D:2D)
        the log std.

    Returns:
     - KL: The Kullback leibler divergence across the batch.
    """

    D = np.shape(q_means_and_log_stds)[-1] // 2
    # Split mean and log std
    mean, log_std = q_means_and_log_stds[:, :D], q_means_and_log_stds[:, D:]

    # Compute the KL divergenve of each element in the batch
    KL = 0.5 * (np.exp(2 * log_std) + mean ** 2 - 1 - 2 * log_std)
    # Sum across the batch
    return np.sum(KL, axis = -1)


def vae_lower_bound(gen_params, rec_params, data):
    """Compute a noisy estimate of the lower bound by using a single
    Monte Carlo sample.

    Arguments:
     - gen_params:  list of (weights, bias) tuples. Containing the parameters
                    of the decoder network. Usable as parameter for
                    neural_net_predict
     - rec_params:  list of (weights, bias) tuples. Containing the parameters
                    of the encoder network. Usable as parameter for
                    neural_net_predict
     - data:        Array-like of shape (batch_size, n_features) containing
                    the data.
    Returns:
        - Noisy estimate of the variational lower bound.
    """

    # Compute the encoder output using neural_net_predict given the data
    # and rec_params
    output = neural_net_predict(params=rec_params, inputs=data)

    # Sample the latent variables associated to the batch in data
    latents = sample_latent_variables_from_posterior(output)

    # Use the sampled latent variables to reconstruct the image and to compute
    # the log_prob of the actual data
    x_samples = neural_net_predict(gen_params, latents)
    log_prob = bernoulli_log_prob(data, x_samples)

    # Compute the KL divergence between q(z|x) and the prior
    KL = compute_KL(output)

    # Return an average estimate (per batch point) of the lower bound by
    # substracting the KL to the data dependent term
    return np.mean(log_prob - KL, axis=-1)


if __name__ == "__main__":

    # Model hyper-parameters
    npr.seed(0)  # We fix the random seed for reproducibility
    latent_dim = 50
    data_dim = 784  # How many pixels in each image (28x28).
    n_units = 200
    n_layers = 2

    gen_layer_sizes = [latent_dim] + [n_units for i in range(n_layers)] \
                    + [data_dim]
    rec_layer_sizes = [data_dim] + [n_units for i in range(n_layers)] \
                    + [latent_dim * 2]

    # Training parameters
    batch_size = 200
    num_epochs = 30
    learning_rate = 0.001

    print("Loading training data...")

    N, train_images, _, test_images, _ = load_mnist()

    print("Done")

    # Parameters for the generator network p(x|z)
    init_gen_params = init_net_params(gen_layer_sizes)

    # Parameters for the recognition network p(z|x)
    init_rec_params = init_net_params(rec_layer_sizes)

    combined_params_init = (init_gen_params, init_rec_params)

    num_batches = int(np.ceil(len(train_images) / batch_size))

    # We flatten the parameters
    flattened_combined_params_init, unflat_params = flatten(combined_params_init)

    # Actual objective to optimize that receives flattened params
    def objective(flattened_combined_params):

        combined_params = unflat_params(flattened_combined_params)
        data_idx = batch
        gen_params, rec_params = combined_params

        # We binarize the data
        on = train_images[data_idx, :] > npr.uniform(
            size=train_images[data_idx, :].shape
        )
        images = train_images[data_idx, :] * 0.0
        images[on] = 1.0

        return vae_lower_bound(gen_params, rec_params, images)

    # Get gradients of objective using autograd.
    objective_grad = grad(objective)
    flattened_current_params = flattened_combined_params_init

    # ADAM parameters
    # the initial values for the ADAM parameters
    # (including the m and v vectors)
    t = 1
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 10**-8
    m = np.zeros_like(flattened_current_params)
    v = np.zeros_like(flattened_current_params)

    # We do the actual training
    for epoch in range(num_epochs):

        elbo_est = 0.0

        for n_batch in range(int(np.ceil(N / batch_size))):

            batch = np.arange(
                batch_size * n_batch, np.minimum(N, batch_size * (n_batch + 1))
            )

            grad = objective_grad(flattened_current_params)

            # Use the estimated noisy gradient in grad to update the parameters
            # using the ADAM updates
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_unbiased = m / (1 - beta1 ** t)
            v_unbiased = v / (1 - beta2 ** t)

            flattened_current_params += (
                alpha * m_unbiased / (np.sqrt(v_unbiased) + epsilon)
            )

            elbo_est += objective(flattened_current_params)

            t += 1

        print("Epoch: %d ELBO: %e" % (epoch, elbo_est / np.ceil(N/batch_size)))

    # We obtain the final trained parameters
    gen_params, rec_params = unflat_params(flattened_current_params)

    # TASK 3.1
    # Generate 25 images from prior (use neural_net_predict) and save
    # them using save_images
    z_samples_prior = npr.randn(25, latent_dim)
    x_samples = neural_net_predict(gen_params, z_samples_prior)
    save_images(sigmoid(x_samples), "images_from_prior")

    # TASK 3.2
    # Generate image reconstructions for the first 10 test images
    # (use neural_net_predict for each model)
    # and save them alongside with the original image using save_images
    output = neural_net_predict(params=rec_params, inputs=test_images[:10])
    latents = sample_latent_variables_from_posterior(output)
    decoding = neural_net_predict(gen_params, latents)
    save_images(sigmoid(decoding), "reconstructions")

    # TASK 3.3

    # Generate 5 interpolations from the first test image to the second
    # test image, for the third to the fourth and so on until 5 interpolations
    # are computed in latent space and save them using save images.
    # To interpolate from  image I to image G use a convex conbination. Namely,
    # I * s + (1-s) * G where s is a sequence of numbers from 0 to 1 obtained
    # by numpy.linspace
    # Use mean of the recognition model as the latent representation.
    num_interpolations = 5

    for i in range(5):
        # Get output of neural network. Output has shape (2D,) given that
        # batch size is "1".
        first_image = neural_net_predict(
            rec_params, test_images[2 * i, :], normalize=False
        )
        second_image = neural_net_predict(
            rec_params, test_images[2 * i + 1, :], normalize=False
        )

        # Get hidden representation from the mean of the recognition model.
        D = np.shape(first_image)[0] // 2
        latents1 = first_image[:D]
        latents2 = second_image[:D]

        # Get interpolation scalars
        S = np.linspace(0, 1, 25)

        # Compute batch of interpolations
        interp = np.array([s * latents1 + (1 - s) * latents2 for s in S])

        # Get image from neural network and plot the result
        image = neural_net_predict(gen_params, interp, normalize=False)
        save_images(sigmoid(image), "interpolation" + str(i))
