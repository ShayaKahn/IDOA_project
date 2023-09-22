import numpy as np

class Overlap:
    """
    This class calculates the overlap value between two given samples
    """
    def __init__(self, sample_first, sample_second, norm=True):
        """
        :param sample_first: first sample, 1D array.
        :param sample_second: second sample, 1D array.
        """
        if not isinstance(sample_first, np.ndarray) or not isinstance(sample_second, np.ndarray):
            raise TypeError("sample_first and sample_second must be numpy arrays")
        if sample_first.shape != sample_second.shape:
            raise ValueError("sample_first and sample_second must have the same length")
        self.sample_first = sample_first
        self.sample_second = sample_second
        if norm:
            self.sample_first = self.sample_first / np.sum(self.sample_first)
            self.sample_second = self.sample_second / np.sum(self.sample_second)
        self.s = self.find_intersection()

    def find_intersection(self):
        """
        This method finds the shared non-zero indexes of the two samples.
        :return: the set s with represent the intersected indexes
        """
        nonzero_index_first = np.nonzero(self.sample_first)  # Find the non-zero index of the first sample.
        nonzero_index_second = np.nonzero(self.sample_second)  # Find the non-zero index of the second sample.
        s = np.intersect1d(nonzero_index_first, nonzero_index_second)  # Find the intersection.
        return s

    def calculate_overlap(self):
        """
        This method calculates the overlap between the two samples.
        :return: the overlap value.
        """
        # Calculation of the overlap value between the two samples.
        overlap = np.sum(self.sample_first[self.s] + self.sample_second[self.s]) / 2
        return overlap

