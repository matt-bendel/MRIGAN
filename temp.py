import matplotlib.pyplot as plt
import numpy as np

from random import randrange


def sampling_mask():
    return 1


def multivariate_gaussian(n, x, mu):
    var = 1
    cov_matrix_inv = 1 / var * np.identity(n)
    diff = x - mu
    exp_arg = diff.T * cov_matrix_inv * diff
    return 1 / np.sqrt((2 * np.pi * var) ** n) * np.exp(exp_arg)


def prior_density(m):
    return 1 / m


def likelihood_density(num_rows, y, i):
    x_i = generate_image(i, num_rows)
    observation = sampling_mask() * np.fft.fft2(x_i)

    return multivariate_gaussian(num_rows**2, y.flatten(), observation.flatten())


def observation_density(m, num_rows, y):
    val = 0
    for i in range(m):
        val += prior_density(m) * likelihood_density(num_rows, y, i)

    return val


def compute_posterior_density(m, num_rows, y):
    posterior = np.zeros(m)
    for i in range(m):
        posterior[i] = prior_density(m) * likelihood_density(num_rows, y, i)

    posterior / observation_density(m, num_rows, y)

    return posterior


def generate_image(i, num_rows):
    square_width = num_rows // 8
    end_row = num_rows - square_width
    horiz_shift = i
    vert_shift = 0

    while horiz_shift > end_row:
        vert_shift += 1
        horiz_shift -= num_rows

    vert_shift = num_rows - vert_shift

    im = np.zeros((num_rows, num_rows))
    im[horiz_shift-1:square_width, vert_shift-1:square_width] = 1

    return im


def main(num_samples):
    for j in range(num_samples):
        num_rows = 64
        square_length = num_rows // 8
        end = num_rows - square_length
        m = end ** 2
        i = randrange(0, m) + 1
        image = generate_image(i, num_rows)
        observation = sampling_mask() * np.fft.fft2(image)
        posterior = compute_posterior_density(m, num_rows, observation)
        i_vals


if __name__ == '__main__':
    main(1)
