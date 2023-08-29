import numpy as np

class Overlap:
    """
    This class calculates the overlap value between two given samples
    """
    def __init__(self, sample_first, sample_second, overlap_type="Overlap"):
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
        [self.normalized_sample_first, self.normalized_sample_second] = self.normalize()
        self.s = self.find_intersection()
        self.overlap_type = overlap_type

    def normalize(self):
        """
        This method normalizes the two samples.
        :return: normalized samples.
        """
        normalized_sample_first = self.sample_first / np.sum(self.sample_first)  # Normalization of the first sample.
        normalized_sample_second = self.sample_second / np.sum(self.sample_second)  # Normalization of the second sample.
        return normalized_sample_first, normalized_sample_second

    def find_intersection(self):
        """
        This method finds the shared non-zero indexes of the two samples.
        :return: the set s with represent the intersected indexes
        """
        nonzero_index_first = np.nonzero(self.normalized_sample_first)  # Find the non-zero index of the first sample.
        nonzero_index_second = np.nonzero(self.normalized_sample_second)  # Find the non-zero index of the second sample.
        s = np.intersect1d(nonzero_index_first, nonzero_index_second)  # Find the intersection.
        return s

    def calculate_overlap(self, non_symmetric=False):
        """
        This method calculates the overlap between the two samples.
        :return: the overlap value.
        """
        # Calculation of the overlap value between the two samples.
        if self.overlap_type == "Overlap":
            if non_symmetric:
                overlap = np.sum(self.normalized_sample_second[self.s])
            else:
                overlap = np.sum(self.normalized_sample_first[self.s] + self.normalized_sample_second[self.s]) / 2
            return overlap
        elif self.overlap_type == "Jaccard":
            smaple_first_boolean = np.where(self.normalized_sample_first != 0, 1, 0)
            smaple_second_boolean = np.where(self.normalized_sample_second != 0, 1, 0)
            intersection = np.sum(np.logical_and(smaple_first_boolean, smaple_second_boolean))
            union = np.sum(np.logical_or(smaple_first_boolean, smaple_second_boolean))
            overlap = intersection / union
            return overlap

