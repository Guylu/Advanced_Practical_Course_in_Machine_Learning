import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.misc
from scipy.special import logsumexp
import pickle
from scipy.stats import multivariate_normal
from skimage.util import view_as_windows as viewW


def images_example(path='train_images.pickle'):
    """
    A function demonstrating how to access to image data supplied in this exercise.
    :param path: The path to the pickle file.
    """
    patch_size = (8, 8)
    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    patches = sample_patches(train_pictures, psize=patch_size, n=20000)

    plt.figure()
    plt.imshow(train_pictures[0])
    plt.title("Picture Example")

    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(patches[:, i].reshape(patch_size), cmap='gray')
        plt.title("Patch Example")
    plt.show()


def im2col(A, window, stepsize=1):
    """
    an im2col function, transferring an image to patches of size window (length
    2 list). the step size is the stride of the sliding window.
    :param A: The original image (NxM size matrix of pixel values).
    :param window: Length 2 list of 2D window size.
    :param stepsize: The step size for choosing patches (default is 1).
    :return: A (heightXwidth)x(NxM) matrix of image patches.
    """
    return viewW(np.ascontiguousarray(A),
                 (window[0], window[1])).reshape(-1, window[0] * window[1]).T[
           :, ::stepsize]


def grayscale_and_standardize(images, remove_mean=True):
    """
    The function receives a list of RGB images and returns the images after
    grayscale, centering (mean 0) and scaling (between -0.5 and 0.5).
    :param images: A list of images before standardisation.
    :param remove_mean: Whether or not to remove the mean (default is True).
    :return: A list of images after standardisation.
    """
    standard_images = []

    for image in images:
        standard_images.append((0.299 * image[:, :, 0] +
                                0.587 * image[:, :, 1] +
                                0.114 * image[:, :, 2]) / 255)

    sum = 0
    pixels = 0
    for image in standard_images:
        sum += np.sum(image)
        pixels += image.shape[0] * image.shape[1]
    dataset_mean_pixel = float(sum) / pixels

    if remove_mean:
        for image in standard_images:
            image -= np.matlib.repmat([dataset_mean_pixel], image.shape[0],
                                      image.shape[1])

    return standard_images


def sample_patches(images, psize=(8, 8), n=10000, remove_mean=True):
    """
    sample N p-sized patches from images after standardising them.

    :param images: a list of pictures (not standardised).
    :param psize: a tuple containing the size of the patches (default is 8x8).
    :param n: number of patches (default is 10000).
    :param remove_mean: whether the mean should be removed (default is True).
    :return: A matrix of n patches from the given images.
    """
    d = psize[0] * psize[1]
    patches = np.zeros((d, n))
    standardized = grayscale_and_standardize(images, remove_mean)

    shapes = []
    for pic in standardized:
        shapes.append(pic.shape)

    rand_pic_num = np.random.randint(0, len(standardized), n)
    rand_x = np.random.rand(n)
    rand_y = np.random.rand(n)

    for i in range(n):
        pic_id = rand_pic_num[i]
        pic_shape = shapes[pic_id]
        x = int(np.ceil(rand_x[i] * (pic_shape[0] - psize[1])))
        y = int(np.ceil(rand_y[i] * (pic_shape[1] - psize[0])))
        patches[:, i] = np.reshape(np.ascontiguousarray(
            standardized[pic_id][x:x + psize[0], y:y + psize[1]]), d)

    return patches


def denoise_image(Y, model, denoise_function, noise_std, patch_size=(8, 8)):
    """
    A function for denoising an image. The function accepts a noisy gray scale
    image, denoises the different patches of it and then reconstructs the image.

    :param Y: the noisy image.
    :param model: a Model object (MVN/ICA/GSM).
    :param denoise_function: a pointer to one of the denoising functions (that corresponds to the model).
    :param noise_std: the noise standard deviation parameter.
    :param patch_size: the size of the patch that the model was trained on (default is 8x8).
    :return: the denoised image, after each patch was denoised. Note, the denoised image is a bit
    smaller than the original one, since we lose the edges when we look at all of the patches
    (this happens during the im2col function).
    """
    (h, w) = np.shape(Y)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))

    # split the image into columns and denoise the columns:
    noisy_patches = im2col(Y, patch_size)
    denoised_patches = denoise_function(noisy_patches, model, noise_std)

    # reshape the denoised columns into a picture:
    x_hat = np.reshape(denoised_patches[middle_linear_index, :],
                       [cropped_h, cropped_w])

    return x_hat


def crop_image(X, patch_size=(8, 8)):
    """
    crop the original image to fit the size of the denoised image.
    :param X: The original picture.
    :param patch_size: The patch size used in the model, to know how much we need to crop.
    :return: The cropped image.
    """
    (h, w) = np.shape(X)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))
    columns = im2col(X, patch_size)
    return np.reshape(columns[middle_linear_index, :], [cropped_h, cropped_w])


def normalize_log_likelihoods(X):
    """
    Given a matrix in log space, return the matrix with normalized columns in
    log space.
    :param X: Matrix in log space to be normalised.
    :return: The matrix after normalization.
    """
    h, w = np.shape(X)
    return X - np.matlib.repmat(logsumexp(X, axis=0), h, 1)


def test_denoising(image, model, denoise_function,
                   noise_range=(
                           0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
                   patch_size=(8, 8)):
    """
    A simple function for testing your denoising code. You can and should
    implement additional tests for your code.
    :param image: An image matrix.
    :param model: A trained model (MVN/ICA/GSM).
    :param denoise_function: The denoise function that corresponds to your model.
    :param noise_range: A tuple containing different noise parameters you wish
            to test your code on. default is (0.01, 0.05, 0.1, 0.2).
    :param patch_size: The size of the patches you've used in your model.
            Default is (8, 8).
    """
    h, w = np.shape(image)
    noisy_images = np.zeros((h, w, len(noise_range)))
    denoised_images = []
    cropped_original = crop_image(image, patch_size)

    # make the image noisy:
    for i in range(len(noise_range)):
        noisy_images[:, :, i] = image + (
                noise_range[i] * np.random.randn(h, w))

    # denoise the image:
    for i in range(len(noise_range)):
        denoised_images.append(
            denoise_image(noisy_images[:, :, i], model, denoise_function,
                          noise_range[i], patch_size))
        print("second: " + str(i))

    # calculate the MSE for each noise range:
    l = []
    for i in range(len(noise_range)):
        # print("noisy MSE for noise = " + str(noise_range[i]) + ":")
        # print(np.mean((crop_image(noisy_images[:, :, i],
        #                           patch_size) - cropped_original) ** 2))
        print("denoised MSE for noise = " + str(noise_range[i]) + ":")
        t = np.mean((cropped_original - denoised_images[i]) ** 2)
        l.append(t)
        print(t)

    plt.plot(noise_range, l)
    for a, b in zip(noise_range, l):
        plt.text(a, b, "%.5f" % round(b, 5), weight='bold')
    plt.title(model.whoami())
    plt.show()

    return noise_range, l

    plt.figure()
    for i in range(len(noise_range)):
        plt.subplot(2, len(noise_range), i + 1)
        plt.imshow(noisy_images[:, :, i], cmap='gray')
        plt.subplot(2, len(noise_range), i + 1 + len(noise_range))
        plt.imshow(denoised_images[i], cmap='gray')
    plt.suptitle(model.whoami())
    plt.show()
    return denoised_images


class MVN_Model:
    """
    A class that represents a Multivariate Gaussian Model, with all the parameters
    needed to specify the model.

    mean - a D sized vector with the mean of the gaussian.
    cov - a D-by-D matrix with the covariance matrix.
    """

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def whoami(self):
        return (type(self).__name__)


class GSM_Model:
    """
    A class that represents a GSM Model, with all the parameters needed to specify
    the model.

    cov - a k-by-D-by-D tensor with the k different covariance matrices. the
        covariance matrices should be scaled versions of each other.
    mix - k-length probability vector for the mixture of the gaussians.
    """

    def __init__(self, cov, mix):
        self.cov = cov
        self.mix = mix

    def whoami(self):
        return (type(self).__name__)


class ICA_Model:
    """
    A class that represents an ICA Model, with all the parameters needed to specify
    the model.

    P - linear transformation of the sources. (X = P*S)
    vars - DxK matrix whose (d,k) element corresponds to the variance of the k'th
        gaussian in the d'th source.
    mix - DxK matrix whose (d,k) element corresponds to the weight of the k'th
        gaussian in d'th source.
    """

    def __init__(self, P, vars, mix):
        self.P = P
        self.vars = vars
        self.mix = mix

    def whoami(self):
        return (type(self).__name__)


def MVN_log_likelihood(X, model):
    """
    Given image patches and a MVN model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A MVN_Model object.
    :return: The log likelihood of all the patches combined.
    """

    sig = model.cov
    mu = model.mean.reshape((X.shape[0], 1))
    det = np.linalg.det(sig)
    inv_sig = np.linalg.inv(sig)
    res = -0.5 * (np.log(det) + (X - mu).T.dot(inv_sig).dot(X - mu) +
                  X.shape[0] * np.log(np.pi * 2))
    return np.sum(res)


def GSM_log_likelihood(X, model):
    """
    Given image patches and a GSM model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A GSM_Model object.
    :return: The log likelihood of all the patches combined.
    """

    n = X.shape[1]
    cov = model.cov
    mix = model.mix
    k = mix.shape[0]

    mle = 0

    for y in range(k):
        mle += mix[y] * multivariate_normal.pdf(X.T, cov=cov[y])
    return np.sum(np.log(mle))


def ICA_log_likelihood(X, model):
    """
    Given image patches and an ICA model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: An ICA_Model object.
    :return: The log likelihood of all the patches combined.
    """
    p = model.P
    vars = model.vars
    mix = model.mix
    S = p.dot(X)
    res = 0
    for s in S:
        res += GSM_log_likelihood(s.reshape(s.shape[0], 1),
                                  GSM_Model(vars, mix))
    return res


def learn_MVN(X):
    """
    Learn a multivariate normal model, given a matrix of image patches.
    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :return: A trained MVN_Model object.
    """

    return MVN_Model(np.mean(X, axis=1), np.cov(X))


def learn_GSM(X, k):
    """
    Learn parameters for a Gaussian Scaling Mixture model for X using EM.

    GSM components share the variance, up to a scaling factor, so we only
    need to learn scaling factors and mixture proportions.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components of the GSM model.
    :return: A trained GSM_Model object.
    """
    # ICA changes:
    if X.shape[0] == 1:
        sig = np.array([[1]])
    else:
        sig = np.cov(X)
    pis = np.random.rand(k).reshape(-1, 1)
    pis = pis / np.sum(pis)
    rs = np.random.rand(k).reshape(-1, 1)
    d = X.shape[0]
    cs = np.zeros(shape=(k, X.shape[1]))
    prev_c = np.zeros(shape=(k, X.shape[1]))
    const_diag = np.sum(X.T.dot(np.linalg.inv(sig)) * X.T, axis=1)
    const_var = 0.5 * (d * np.log(2 * np.pi) + np.log(np.linalg.det(sig)))
    basic_tile = np.tile(const_diag, (k, 1))
    while not np.all(np.isclose(cs, prev_c)):
        prev_c = cs.copy()
        terms = np.log(pis) - const_var
        tiles = basic_tile * (1 / (2 * rs))
        pre_cs = terms - tiles
        cs = np.exp(normalize_log_likelihoods(pre_cs))
        pis = np.mean(cs, axis=1).reshape(-1, 1)
        rs = (cs.dot(const_diag) / (d * np.sum(cs, axis=1))).reshape(-1, 1)

    return GSM_Model(np.tile(sig, (k, 1, 1)) * rs.reshape(k, 1, 1), pis)


def learn_ICA(X, k):
    """
    Learn parameters for a complete invertible ICA model.

    We learn a matrix P such that X = P*S, where S are D independent sources
    And for each of the D coordinates we learn a mixture of K univariate
    0-mean gaussians using EM.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components in the source gaussian mixtures.
    :return: A trained ICA_Model object.
    """

    """
    plan: 
    To learn the parameters of the d mixtures of k uni-variate Gaussians, we perform the following
    learning procedure:
    1. Calculate P from our training set by diagonalizing the empirical covariance matrix.
    2. Transform our training set to the s domain: s = P T x
    3. Learn each of the d coordinates in the s domain separately using the EM algorithm, assuming
    each coordinate is sampled from a mixture of k uni-variate Gaussians
    """

    # 1:
    p = np.linalg.svd(np.cov(X))[0]
    s = p.T.dot(X)
    d = s.shape[0]
    vars = np.zeros(shape=(d, k))
    mix = np.zeros(shape=(d, k))

    for i in range(d):
        model = learn_GSM(s[i].reshape((1, s.shape[1])), k)
        vars[i], mix[i] = np.squeeze(model.cov), np.squeeze(model.mix)
    return ICA_Model(p, vars, mix)


def MVN_Denoise(Y, mvn_model, noise_std):
    """
    Denoise every column in Y, assuming an MVN model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a single
    0-mean multi-variate normal distribution.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param mvn_model: The MVN_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """

    inv_sig = np.linalg.inv(mvn_model.cov)
    post_weiner = np.linalg.inv(inv_sig + np.eye(inv_sig.shape[0]) *
                                (1 / (noise_std ** 2)))
    post_weiner = post_weiner.dot(Y / (noise_std ** 2))

    return post_weiner


def GSM_Denoise(Y, gsm_model, noise_std):
    """
    Denoise every column in Y, assuming a GSM model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a mixture of
    0-mean gaussian components sharing the same covariance up to a scaling factor.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param gsm_model: The GSM_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.

    """

    sigs = gsm_model.cov
    # simple fix for my code
    sigs = sigs.T
    # ICA change:
    if sigs.ndim == 1:
        sigs = np.expand_dims(sigs, axis=(0, 1))
    mix = gsm_model.mix
    k = mix.shape[0]
    cs = np.zeros(shape=(k, Y.shape[1]))
    res = np.zeros_like(Y)

    for i in range(k):
        cs[i] = np.log(mix[i])
        new_sig = sigs[:, :, i] + (np.eye(sigs.shape[0]) * (noise_std ** 2))
        const_term = np.linalg.det(new_sig * (2 * np.pi))
        cs[i] += np.log(1 / np.sqrt(const_term))
        cs[i] -= 0.5 * np.log(const_term)
        product = (Y.T.dot(np.linalg.inv(new_sig)) * Y.T).sum(-1)
        cs[i] -= product / 2
    cs = np.exp(normalize_log_likelihoods(cs))

    for i in range(k):
        inv_sig = np.linalg.inv(sigs.T[i])
        post_weiner = np.linalg.inv(inv_sig + np.eye(inv_sig.shape[0]) *
                                    (1 / (noise_std ** 2)))
        post_weiner = post_weiner.dot(Y / (noise_std ** 2))
        res += post_weiner * cs[i]

    return res


def ICA_Denoise(Y, ica_model, noise_std):
    """
    Denoise every column in Y, assuming an ICA mode l and gaussian white noise.

    The model assumes that y = x + noise where x is generated by an ICA 0-mean
    mixture model.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param ica_model: The ICA_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """

    """
    1. Move y to the s domain: s_noisy = P T y.
    2. Denoise each of the coordinates of s_noisy separately using the Wiener
     filter to get s_denoised.
    3. Move back to the image patch domain to get our denoised image: 
    x_denoised = P ?? s_denoised
    """
    p = ica_model.P
    vars = ica_model.vars
    mix = ica_model.mix
    x_denoised = np.zeros_like(Y)

    s_noisy = p.T.dot(Y)
    d = s_noisy.shape[0]
    for i in range(d):
        s_denoised = GSM_Denoise(s_noisy[i].reshape(1, s_noisy.shape[1]),
                                 GSM_Model(vars[i], mix[i]), noise_std)
        x_denoised[i] = s_denoised
    return p.dot(x_denoised)


if __name__ == '__main__':
    patch_size = (8, 8)
    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    patches = sample_patches(train_pictures, psize=patch_size, n=20000)
    im_num = 2
    mvn_model = learn_MVN(patches)
    grayscales = grayscale_and_standardize(train_pictures)
    a1 = test_denoising(grayscales[im_num], mvn_model, MVN_Denoise)

    
    test_denoising(a1[2], mvn_model, MVN_Denoise,
                   noise_range=(0.000001, 0.000001))

    for k in [1, 2, 3, 5, 10, 20, 100]:
        gsm_model = learn_GSM(patches, k)
        grayscales = grayscale_and_standardize(train_pictures)
        a2 = test_denoising(grayscales[im_num], gsm_model, GSM_Denoise)

        ica_model = learn_ICA(patches, k)
        grayscales = grayscale_and_standardize(train_pictures)
        a3 = test_denoising(grayscales[im_num], ica_model, ICA_Denoise)

    mvn_model = learn_MVN(patches)
    print("MVN log likelihood: " +
          str(MVN_log_likelihood(patches, mvn_model)))

    gsm_model = learn_GSM(patches, 5)
    print("GSM log likelihood: " +
          str(GSM_log_likelihood(patches, gsm_model)))

    ica_model = learn_ICA(patches, 5)
    print("ICA log likelihood: " +
          str(ICA_log_likelihood(patches, ica_model)))
