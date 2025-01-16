import numpy as np

def multivariate_gaussian(x, mean, cov):
    """
    Computes the multivariate Gaussian probability density function (PDF).

    Parameters:
    - x (array): Data point (1D array, shape: [d])
    - mean (array): Mean of the Gaussian (shape: [d])
    - cov (array): Covariance matrix (shape: [d, d])

    Returns:
    - float: Probability density for the data point under the given Gaussian.
    """
    d = len(mean)  # Dimensionality of the data
    epsilon = 1e-6  # Small value to ensure numerical stability
    cov += np.eye(d) * epsilon  # Add small value to diagonal

    # Compute normalization constant
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    norm_const = 1.0 / (np.sqrt((2 * np.pi) ** d * det_cov))

    # Compute Mahalanobis distance
    x_mu = x - mean
    mahalanobis_dist = np.dot(np.dot(x_mu.T, inv_cov), x_mu)

    # Compute Gaussian PDF
    pdf = norm_const * np.exp(-0.5 * mahalanobis_dist)
    return pdf


def e_step(x, mean, covariance, mixing_coefficients, gaussian_probs):
    """
    Compute responsibilities using precomputed Gaussian probabilities.

    Parameters:
    - x (array): Data points (shape: [n_samples, d])
    - mean (array): Means of Gaussian components (shape: [n_clusters, d])
    - covariance (array): Covariance matrices of components (shape: [n_clusters, d, d])
    - mixing_coefficients (array): Mixing coefficients (shape: [n_clusters])
    - gaussian_probs (array): Precomputed probabilities (shape: [n_samples, n_clusters])

    Returns:
    - array: Responsibilities for each data point (shape: [n_samples, n_clusters]).
    """
    n_samples, n_clusters = x.shape[0], len(mean)

    responsibilities = np.zeros((n_samples, n_clusters))
    for i in range(n_samples):
        denominator = np.sum([mixing_coefficients[m] * gaussian_probs[i, m] for m in range(n_clusters)])
        for k in range(n_clusters):
            responsibilities[i, k] = (mixing_coefficients[k] * gaussian_probs[i, k]) / denominator

    return responsibilities


def m_step(x, responsibilities):
    """
    M-step of the EM algorithm: Update the parameters (means, covariances, mixing coefficients).

    Parameters:
    - x (array): Data points (shape: [n_samples, d])
    - responsibilities (array): Responsibilities from E-step (shape: [n_samples, n_clusters])

    Returns:
    - tuple: Updated means, covariances, and mixing coefficients.
    """
    n_samples, n_features = x.shape
    n_clusters = responsibilities.shape[1]

    # Update mixing coefficients
    Nk = responsibilities.sum(axis=0)  # Total responsibilities for each cluster
    mixing_coefficients = Nk / n_samples

    # Update means
    means = np.dot(responsibilities.T, x) / Nk[:, None]

    # Update covariances
    covariances = np.zeros((n_clusters, n_features, n_features))
    for k in range(n_clusters):
        x_centered = x - means[k]
        weighted_cov = np.dot(
            (responsibilities[:, k][:, None] * x_centered).T, x_centered
        )
        covariances[k] = weighted_cov / Nk[k]

    return means, covariances, mixing_coefficients


def log_likelihood(x, means, covariances, mixing_coefficients, gaussian_probs=None):
    """
    Compute the log-likelihood for the current iteration of the EM algorithm.
    
    Parameters:
    - x: Data points (shape: [n_samples, d])
    - means: Means of Gaussian components (shape: [n_clusters, d])
    - covariances: Covariance matrices of components (shape: [n_clusters, d, d])
    - mixing_coefficients: Mixing coefficients (shape: [n_clusters])
    - gaussian_probs: (Optional) Precomputed probabilities (shape: [n_samples, n_clusters])
    
    Returns:
    - log_likelihood_value: Log-likelihood value.
    """
    n_samples = x.shape[0]
    n_clusters = len(means)

    # If gaussian_probs is None, compute it
    if gaussian_probs is None:
        gaussian_probs = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            for i in range(n_samples):
                gaussian_probs[i, k] = multivariate_gaussian(x[i], means[k], covariances[k])

    # Compute log-likelihood
    log_likelihood_value = 0.0
    for i in range(n_samples):
        weighted_sum = 0.0
        for k in range(n_clusters):
            weighted_sum += mixing_coefficients[k] * gaussian_probs[i, k]
        log_likelihood_value += np.log(weighted_sum)

    return log_likelihood_value


def em_algorithm(x, means, covariances, mixing_coefficients, max_iters=100, tol=1e-4):
    """
    Perform the EM algorithm iteratively until convergence.

    Parameters:
    - x (array): Data points (shape: [n_samples, d])
    - means (array): Initial means (shape: [n_clusters, d])
    - covariances (array): Initial covariances (shape: [n_clusters, d, d])
    - mixing_coefficients (array): Initial mixing coefficients (shape: [n_clusters])
    - max_iters (int): Maximum number of iterations (default: 100)
    - tol (float): Convergence threshold for log-likelihood (default: 1e-4)

    Returns:
    - tuple: Final means, covariances, mixing coefficients, and log-likelihoods.
    """
    log_likelihoods = []

    for iteration in range(max_iters):
        # E-step
        responsibilities = e_step(x, means, covariances, mixing_coefficients)

        # M-step
        means, covariances, mixing_coefficients = m_step(x, responsibilities)

        # Log-likelihood
        log_likelihood_value = log_likelihood(x, means, covariances, mixing_coefficients)
        log_likelihoods.append(log_likelihood_value)
        print(f"Iteration {iteration + 1}, Log-Likelihood: {log_likelihood_value}")

        # Check for convergence
        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            print("Convergence reached.")
            break

    return means, covariances, mixing_coefficients, log_likelihoods


def assign_clusters(x, means, covariances, mixing_coefficients):
    """
    Assign data points to clusters based on maximum posterior probabilities.

    Parameters:
    - x (array): Data points (shape: [n_samples, d])
    - means (array): Means of Gaussian components (shape: [n_clusters, d])
    - covariances (array): Covariance matrices (shape: [n_clusters, d, d])
    - mixing_coefficients (array): Mixing coefficients (shape: [n_clusters])

    Returns:
    - array: Cluster assignments (shape: [n_samples]).
    """
    responsibilities = e_step(x, means, covariances, mixing_coefficients)
    return np.argmax(responsibilities, axis=1)