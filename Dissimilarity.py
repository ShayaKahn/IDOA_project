import numpy as np
from scipy.spatial.distance import braycurtis, euclidean

class Dissimilarity:
    """
    This class calculates the dissimilarity value between two given samples.
    """
    def __init__(self, sample_first, sample_second, dissimilarity_type="rjsd"):
        """
        Arguments:
        sample_first -- first sample, 1D array of the optional shapes: (n_features,), (1,n_features), (n_features,1).
        sample_second -- second sample, 1D array of the optional shapes: (n_features,), (1,n_features), (n_features,1).
        dissimilarity_type -- The type of the dissimilarity, optional values are:
                              rjsd, jsd, BC, Euclidean. Type: string. Default: rjsd.
        """
        if not isinstance(dissimilarity_type, str) or dissimilarity_type not in ["rjsd", "jsd", "BC", "euclidean"]:
            raise TypeError("dissimilarity_type must be a string with optional values:"
                            " 'rjsd', 'jsd', 'BC', 'euclidean'")
        self.dissimilarity_type = dissimilarity_type
        if not isinstance(sample_first, np.ndarray) or not isinstance(sample_second, np.ndarray):
            raise TypeError("sample_first and sample_second must be numpy arrays")
        if sample_first.ndim == 2:
            self.sample_first = sample_first.reshape(-1)
        else:
            self.sample_first = sample_first
        if sample_second.ndim == 2:
            self.sample_second = sample_second.reshape(-1)
        else:
            self.sample_second = sample_second
        assert self.sample_first.size == self.sample_second.size, 'The number of features must be equal!'
        self.normalized_sample_first, self.normalized_sample_second = self.normalize()
        self.s = self.find_intersection()
        self.normalized_sample_first_hat, self.normalized_sample_second_hat, self.z = self.calculate_normalized_in_s()

    def normalize(self):
        """
        This method normalizes the two samples.

        Return:
        normalized_sample_first, normalized_sample_second -- The two normalized samples.
        """
        normalized_sample_first = self.sample_first / np.sum(self.sample_first)  # Normalization of the first sample.
        normalized_sample_second = self.sample_second / np.sum(self.sample_second)  # Normalization of the second sample.
        return normalized_sample_first, normalized_sample_second

    def find_intersection(self):
        """
        This method finds the shared non-zero indexes of the two samples.

        Return:
        s -- the set s with represent the intersected indexes.
        """
        nonzero_index_first = np.nonzero(self.normalized_sample_first)  # Find the non-zero index of the first sample.
        nonzero_index_second = np.nonzero(self.normalized_sample_second)  # Find the non-zero index of the second sample.
        s = np.intersect1d(nonzero_index_first, nonzero_index_second)  # Find the intersection.
        return s

    def calculate_normalized_in_s(self):
        """
        This method calculates the normalized samples inside s.

        Return:
        normalized_sample_first_hat, normalized_sample_second_hat, z -- the normalized samples inside s and variable z.
        """
        normalized_sample_first_hat = self.normalized_sample_first[self.s] /\
                                      np.sum(self.normalized_sample_first[self.s])
        normalized_sample_second_hat = self.normalized_sample_second[self.s] /\
                                       np.sum(self.normalized_sample_second[self.s])
        z = (normalized_sample_first_hat + normalized_sample_second_hat) / 2  # define z variable
        return normalized_sample_first_hat, normalized_sample_second_hat, z

    def dkl(self, u_hat):
        """
        This method calculates the Kulleback-Leibler divergence.

        Arguments:
        u_hat -- 1D vector.
        Return:
        dkl value.
        """
        return np.sum(u_hat*np.log(u_hat/self.z))  # Calculate dkl

    def calculate_dissimilarity(self):
        """
        This method calculates the dissimilarity value.

        Return:
        dissimilarity value.
        """
        # Calculate dissimilarity
        if self.dissimilarity_type == "rjsd":
            sqrt_value = (self.dkl(self.normalized_sample_first_hat) +
                          self.dkl(self.normalized_sample_second_hat)) / 2

            # Check if the value inside the sqrt is negative
            if sqrt_value < 0:
                dissimilarity = 0
            else:
                dissimilarity = np.sqrt(sqrt_value)


            return dissimilarity
        elif self.dissimilarity_type == "jsd":
            dissimilarity = (self.dkl(self.normalized_sample_first_hat) +
                             self.dkl(self.normalized_sample_second_hat)) / 2
            return dissimilarity
        elif self.dissimilarity_type == "BC":
            dissimilarity = braycurtis(self.normalized_sample_first_hat,
                                       self.normalized_sample_second_hat)
            return dissimilarity
        elif self.dissimilarity_type == "euclidean":
            dissimilarity = euclidean(self.normalized_sample_first_hat,
                                       self.normalized_sample_second_hat)
            return dissimilarity
