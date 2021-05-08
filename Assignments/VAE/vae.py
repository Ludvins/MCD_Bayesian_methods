# Implements a variational auto-encoder for the MNIST problem.

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

from autograd import grad
from data import load_mnist
from data import save_images as s_images
from autograd.misc import (
    flatten,
)  # This is used to flatten the params (transforms a list into a numpy array)


def save_images(images, file_name=None):
    """Calls s_images with fixed parameters:
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
    """This function intializes the parameters of a deep neural network.
    Returns an array of (weights, biases) for each layer.
    """
    return [
        (scale * npr.randn(m, n), scale * npr.randn(n))  # (weight matrix, bias vector)
        for m, n in zip(layer_sizes[:-1], layer_sizes[1:])
    ]


# This will be used to normalize the activations of the NN


def neural_net_predict(params, inputs):
    """This computes the output of a deep neural network with params a
    list with pairs of weights and biases

    Arguments:
     - Params: list of (weights, bias) tuples.
     - inputs: (N x D) matrix.

    Applies batch normalization to every layer but the last.
    """

    for W, b in params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)  # nonlinear transformation

    # Last layer is linear
    outW, outb = params[-1]
    outputs = np.dot(inputs, outW) + outb

    return outputs


def sample_latent_variables_from_posterior(encoder_output):
    """Generates a sample from q(z|x) per each batch datapoint
    using the reparameterization trick
    """
    # Number of outputs
    D = np.shape(encoder_output)[-1] // 2
    # The first half corresponds to the mean and the second to the log std
    mean, log_std = encoder_output[:, :D], encoder_output[:, D:]

    # TODO use the reparametrization trick to generate one sample from q(z|x) per each batch datapoint
    # use npr.randn for that.
    # The output of this function is a matrix of size the batch x the number of latent dimensions
    z = npr.randn(*mean.shape)
    z = mean + z * np.exp(log_std)

    return z


def bernoulli_log_prob(targets, logits):
    """
    Computes the log probability of the targets given the generator output
    specified in logits

    Arguments:
     - logits. Real values. Unnormalized log-probs
     - targets. Values in [0, 1].

    Returns
     - log_prob: vector of size batch_size
    """
    # Opcion 1
    probs = sigmoid(logits)
    log_prob = np.log(targets * probs + (1 - targets) * (1 - probs))
    # Opcion 2 ( no tengo claro cual hacer)
    log_prob = -np.logaddexp(0, -logits * targets)

    # Sum the probabilities across the dimensions of each image in the batch.
    return np.sum(log_prob, axis=-1)


def compute_KL(q_means_and_log_stds):
    """
    Compute the KL divergence between q(z|x) and the prior (use a standard Gaussian for the prior.
    Use the fact that the KL divervence is the sum of KL divergence of the marginals if q and p factorize
    """

    D = np.shape(q_means_and_log_stds)[-1] // 2
    mean, log_std = q_means_and_log_stds[:, :D], q_means_and_log_stds[:, D:]

    # The output of this function should be a vector of size the batch size
    KL = 0.5 * (np.exp(log_std) + mean ** 2 - 1 - log_std)
    return np.sum(KL, axis=-1)


def generate_from_prior(gen_params, num_samples, noise_dim, rs):
    latents = rs.randn(num_samples, noise_dim)
    return sigmoid(neural_net_predict(gen_params, latents))


def vae_lower_bound(gen_params, rec_params, data):
    """Compute a noisy estimate of the lower bound by using a single Monte Carlo sample."""

    # Compute the encoder output using neural_net_predict given the data and rec_params
    output = neural_net_predict(params=rec_params, inputs=data)

    # Sample the latent variables associated to the batch in data
    #     (use sample_latent_variables_from_posterior and the encoder output)
    latents = sample_latent_variables_from_posterior(output)

    # Use the sampled latent variables to reconstruct the image and to compute the log_prob of the actual data
    #     (use neural_net_predict for that)
    x_samples = neural_net_predict(gen_params, latents)
    log_prob = bernoulli_log_prob(data, x_samples)

    # Compute the KL divergence between q(z|x) and the prior (use compute_KL for that)
    KL = compute_KL(output)

    # Return an average estimate (per batch point) of the lower bound by substracting the KL to the data dependent term
    return np.mean(KL - log_prob, axis=-1)


if __name__ == "__main__":

    # Model hyper-parameters
    npr.seed(0)  # We fix the random seed for reproducibility
    latent_dim = 50
    data_dim = 784  # How many pixels in each image (28x28).
    n_units = 200
    n_layers = 2

    gen_layer_sizes = [latent_dim] + [n_units for i in range(n_layers)] + [data_dim]
    rec_layer_sizes = [data_dim] + [n_units for i in range(n_layers)] + [latent_dim * 2]

    # Training parameters
    batch_size = 200
    num_epochs = 30
    learning_rate = 0.001

    print("Loading training data...", end=" ")

    N, train_images, _, test_images, _ = load_mnist(sklearn=True)

    print("Done")

    # Parameters for the generator network p(x|z)
    init_gen_params = init_net_params(gen_layer_sizes)

    # Parameters for the recognition network p(z|x)
    init_rec_params = init_net_params(rec_layer_sizes)

    combined_params_init = (init_gen_params, init_rec_params)

    num_batches = int(np.ceil(len(train_images) / batch_size))

    # We flatten the parameters (transform the lists or tupples into numpy arrays)
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

        return -vae_lower_bound(gen_params, rec_params, images)

    # Get gradients of objective using autograd.
    objective_grad = grad(objective)
    flattened_current_params = flattened_combined_params_init

    # ADAM parameters
    # the initial values for the ADAM parameters (including the m and v vectors)
    t = 1
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 10 ** -8
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

            # Use the estimated noisy gradient in grad to update the parameters using the ADAM updates
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_unbiased = m / (1 - beta1 ** t)
            v_unbiased = v / (1 - beta2 ** t)

            flattened_current_params = flattened_current_params + alpha * m_unbiased / (
                np.sqrt(v_unbiased) + epsilon
            )

            elbo_est += objective(flattened_current_params)

            t += 1

        print("Epoch: %d ELBO: %e" % (epoch, elbo_est / np.ceil(N / batch_size)))

    # We obtain the final trained parameters

    gen_params, rec_params = unflat_params(flattened_current_params)

    # TODO Generate 25 images from prior (use neural_net_predict) and save them using save_images
    z_samples_prior = npr.randn(25, latent_dim)
    x_samples = neural_net_predict(gen_params, z_samples_prior)
    save_images(x_samples)

    # TODO Generate image reconstructions for the first 10 test images (use neural_net_predict for each model)
    # and save them alongside with the original image using save_images

    num_interpolations = 5
    for i in range(5):

        # TODO Generate 5 interpolations from the first test image to the second test image,
        # for the third to the fourth and so on until 5 interpolations
        # are computed in latent space and save them using save images.
        # Use a different file name to store the images of each iterpolation.
        # To interpolate from  image I to image G use a convex conbination. Namely,
        # I * s + (1-s) * G where s is a sequence of numbers from 0 to 1 obtained by numpy.linspace
        # Use mean of the recognition model as the latent representation.

        pass
