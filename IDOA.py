import numpy as np
from Overlap import Overlap
from Dissimilarity import Dissimilarity
from scipy.stats import linregress

class IDOA:
    """
    This class calculates the IDOA values vector for a cohort or the IDOA value for a sample
    with respect to a reference cohort.
    """
    def __init__(self, ref_cohort, cohort, identical=False, percentage=50, min_num_points=0, min_overlap=0.5,
                 max_overlap=1, zero_overlap=0.1, method='percentage'):
        """
        :param ref_cohort: The reference cohort, samples are in the rows.
        :param cohort: The cohort, samples are in the rows.
        :param identical: If True, both cohorts are considered as identical.
        :param percentage: The percentage of the lower overlap points to exclude from IDOA calculation.
        :param min_num_points: The minimal number of points to calculate the IDOA.
        :param min_overlap: The minimal value of overlap.
        :param max_overlap: The maximal value of overlap.
        :param zero_overlap: A number, if the maximal value of the overlap vector that calculated
               between sample from the second cohort w.r.t the first cohort is less than min_overlap + zero_overlap
               so the overlap considered to be zero.
        :param method: the method to define the overlap range for IDOA clculation, optional values are: 'percentage' and
                       'min_max_zero'.
        """
        self.ref_cohort = ref_cohort
        self.cohort = cohort
        self.min_num_points = min_num_points
        if type(self.min_num_points) is not int:
            raise ValueError("min_num_points should be a positive integer")
        if self.ref_cohort.ndim != 2:
            raise ValueError("ref_cohort should be a 2D numpy array")
        if self.cohort.ndim not in [1, 2]:
            raise ValueError("cohort should be a 1D or 2D numpy array")
        if self.cohort.ndim == 1 and self.cohort.size != self.ref_cohort.shape[1]:
            raise ValueError("The size of cohort should match the number of columns in ref_cohort")
        if self.cohort.ndim == 2 and self.cohort.shape[1] != self.ref_cohort.shape[1]:
            raise ValueError("The number of columns in cohort should match the number of columns in ref_cohort")
        try:
            self.min_overlap = min_overlap
            assert isinstance(self.min_overlap, (int, float))
            self.max_overlap = max_overlap
            assert isinstance(self.max_overlap, (int, float))
            self.zero_overlap = zero_overlap
            assert isinstance(self.zero_overlap, (int, float))
        except AssertionError:
            raise ValueError("Invalid input values for min_overlap, max_overlap or zero_overlap. "
                             "They should be numeric values")
        try:
            assert (0 <= min_overlap < 1)
            assert (min_overlap < max_overlap and (0 < max_overlap <= 1))
            assert (min_overlap + zero_overlap < max_overlap)
        except AssertionError:
            raise ValueError("Invalid input values for min_overlap, max_overlap or zero_overlap. "
                             "Their values should be between 0 and 1, with min_overlap less than max_overlap "
                             "and min_overlap plus zero_overlap less than max_overlap.")
        self.identical = identical
        try:
            assert isinstance(self.identical, bool)
        except AssertionError:
            raise AssertionError("Invalid input values for identical, should be boolean value.")
        if self.identical and self.ref_cohort.shape != self.cohort.shape:
            raise ValueError("If identical=True, the dimensions of self.cohort and self.ref_cohort should be the same.")
        self.percentage = percentage
        if self.percentage:
            if not (isinstance(self.percentage, int) or isinstance(self.percentage, float)):
                raise ValueError('percentage should be int or float')
            if not (0 <= self.percentage <= 100):
                raise ValueError('percentage must take values between 0 and 100 inclusive.')
        self.method = method
        if not (self.method == 'percentage' or self.method == 'min_max_zero'):
            raise ValueError('Optional values for method are percentage or min_max_zero')
        self.num_samples_ref = ref_cohort.shape[0]
        self.num_samples_cohort = cohort.shape[0]
        self.IDOA_vector = 0 if self.cohort.ndim == 1 else np.zeros(self.num_samples_cohort)
        self.dissimilarity_overlap_container = []
        self.dissimilarity_overlap_container_no_constraint = []

    def _create_od_vectors(self, sample, index=(None,)):
        """
        :param sample: A sample
        :param index: Integer
        :return: overlap_vector and dissimilarity_vector, this vectors contain the overlap and dissimilarity values.
        """
        o_vector = []
        d_vector = []
        for j in range(0, self.num_samples_ref):
            # verify that samples are not identical
            if not np.array_equal(self.ref_cohort[j, :], sample):
                o = Overlap(self.ref_cohort[j, :], sample)  # Initiate Overlap
                d = Dissimilarity(self.ref_cohort[j, :], sample)  # Initiate Dissimilarity
            else:
                o = None
                d = None
            o_vector.append(o)
            d_vector.append(d)
        # Calculate overlap values
        overlap_vector = np.array([o_vector[j].calculate_overlap()
                                   for j in range(0, self.num_samples_ref) if not (j == index or o_vector[j] is None)])
        # Calculate dissimilarity values
        dissimilarity_vector = np.array([d_vector[j].calculate_dissimilarity()
                                        for j in range(0, self.num_samples_ref) if not (j == index or
                                                                                        o_vector[j] is None)])
        # dissimilarity vector
        return overlap_vector, dissimilarity_vector

    def _filter_od_vectors(self, overlap_vector, dissimilarity_vector):
        """
        :param overlap_vector: Vector that contains the overlap values
        :param dissimilarity_vector: Vector that contains the dissimilarity values
        :return: filtered_overlap_vector and filtered_dissimilarity_vector, the original vectors after filtering.
        """
        if self.method == 'percentage':
            overlap_vector_index = np.where(overlap_vector > np.percentile(overlap_vector, self.percentage))
        else:
            overlap_vector_index = np.where(np.logical_and(overlap_vector >= self.min_overlap,
                                                           overlap_vector <= self.max_overlap))
            if overlap_vector_index[0].size == 0:
                raise ValueError("No overlap values found within the given range")
        filtered_overlap_vector = overlap_vector[overlap_vector_index]
        filtered_dissimilarity_vector = dissimilarity_vector[overlap_vector_index]
        return filtered_overlap_vector, filtered_dissimilarity_vector

    def _calc_idoa_vector_sample_vs_cohort(self, ind):
        """
        This is a private method that calculates the IDOA value for single sample w.r.t a cohort
        :param ind: index
        :return: IDOA vector
        """
        if ind[0] is not None:
            # Check if variable is a tuple
            if not isinstance(ind, tuple):
                raise ValueError("Invalid input value for ind, should be a tuple")

            # Check if the tuple has exactly one element
            if len(ind) != 1:
                raise ValueError("Invalid input value for ind, should be a tuple with exactly one element")

            # Check if the single element is an integer
            if not isinstance(ind[0], int):
                raise ValueError("Invalid input value for ind, should be a tuple with exactly one integer element")

        if ind[0] is not None:
            k = ind
            overlap_vector, dissimilarity_vector = self._create_od_vectors(self.cohort, index=k)
        else:
            overlap_vector, dissimilarity_vector = self._create_od_vectors(self.cohort)
        self.dissimilarity_overlap_container_no_constraint.append(np.vstack((overlap_vector, dissimilarity_vector)))
        # Set IDOA as 0 for low overlap values
        if np.max(overlap_vector) <= self.min_overlap + self.zero_overlap and not self.percentage:
            self.IDOA_vector = 0
        else:
            filtered_overlap_vector, filtered_dissimilarity_vector = self._filter_od_vectors(
                overlap_vector, dissimilarity_vector)
            self.dissimilarity_overlap_container.append(np.vstack((filtered_overlap_vector,
                                                                   filtered_dissimilarity_vector)))
            slope = linregress(filtered_overlap_vector, filtered_dissimilarity_vector)[0]  # Calculate the slope
            self.IDOA_vector = slope if not np.isnan(slope) else 0  # If the slope is a valid
            # number set: IDOA = slope
            if np.size(filtered_overlap_vector) < self.min_num_points:
                self.IDOA_vector = 0
            return self.IDOA_vector

    def _calc_idoa_vector_cohort_vs_cohort_identical(self):
        """
        This is a private method that calculates the IDOA vector for a cohort w.r.t to itself
        :return: IDOA vector
        """
        for i in range(0, self.num_samples_cohort):
            overlap_vector, dissimilarity_vector = self._create_od_vectors(
                self.cohort[i, :], index=[i])
            self.dissimilarity_overlap_container_no_constraint.append(np.vstack((overlap_vector,
                                                                                dissimilarity_vector)))
             # Set IDOA as 0 for low overlap values
            if np.max(overlap_vector) <= self.min_overlap + self.zero_overlap and not self.percentage:
                self.IDOA_vector[i] = 0
            else:
                filtered_overlap_vector, filtered_dissimilarity_vector = self._filter_od_vectors(
                    overlap_vector, dissimilarity_vector)
                self.dissimilarity_overlap_container.append(np.vstack((filtered_overlap_vector,
                                                                       filtered_dissimilarity_vector)))
                slope = linregress(filtered_overlap_vector, filtered_dissimilarity_vector)[0]  # Calculate the slope
                self.IDOA_vector[i] = slope if not np.isnan(slope) else 0  # If the slope is a valid number set:
                # IDOA = slope
                if np.size(filtered_overlap_vector) < self.min_num_points:
                    self.IDOA_vector[i] = 0
        return self.IDOA_vector

    def _calc_idoa_vector_cohort_vs_cohort_custom(self, ind):
        """
        This is a private method that calculates the IDOA vector for a cohort w.r.t to a different cohort and includes
        constraints on specific indexes
        :param ind: dictionary of indexes, if i --> j it means that the dissimilarity and overlap are not calculated for
         sample i w.r.t sample j
        :return: IDOA vector
        """
        if not isinstance(ind, tuple):
            # Check if ind is a dictionary
            if not isinstance(ind, dict):
                raise TypeError("ind must be a dictionary")

            # Iterate through the items in the dictionary
            for key, value in ind.items():
                # Check if key is an integer
                if not isinstance(key, int):
                    raise TypeError("all keys in ind must be an integers")

                # Check if value is a tuple
                if not isinstance(value, tuple):
                    raise TypeError("all values in ind must be a tuples")

                # Check if all elements in the tuple are integers
                if not all(isinstance(element, int) for element in value):
                    raise TypeError("all the elements the values must contain only integers")

        for i in range(0, self.num_samples_cohort):
            if i in ind:
                k = ind[i]
                overlap_vector, dissimilarity_vector = self._create_od_vectors(
                    self.cohort[i, :], index=k)
            else:
                overlap_vector, dissimilarity_vector = self._create_od_vectors(
                    self.cohort[i, :])
            self.dissimilarity_overlap_container_no_constraint.append(np.vstack((overlap_vector,
                                                                                dissimilarity_vector)))
            # Set IDOA as 0 for low overlap values
            if np.max(overlap_vector) <= self.min_overlap + self.zero_overlap and not self.percentage:
                self.IDOA_vector[i] = 0
            else:
                filtered_overlap_vector, filtered_dissimilarity_vector = self._filter_od_vectors(
                    overlap_vector, dissimilarity_vector)
                self.dissimilarity_overlap_container.append(np.vstack((filtered_overlap_vector,
                                                                       filtered_dissimilarity_vector)))
                slope = linregress(filtered_overlap_vector, filtered_dissimilarity_vector)[0]  # Calculate the slope
                self.IDOA_vector[i] = slope if not np.isnan(slope) else 0  # If the slope is a valid number set:
                # IDOA = slope
                if np.size(filtered_overlap_vector) < self.min_num_points:
                    self.IDOA_vector[i] = 0
        return self.IDOA_vector

    def _calc_idoa_vector_cohort_vs_cohort_not_identical(self):
        """
        This is a private method that calculates the IDOA vector for a cohort w.r.t to a different cohort
        :return: IDOA
        """
        for i in range(0, self.num_samples_cohort):
            overlap_vector, dissimilarity_vector = self._create_od_vectors(
                self.cohort[i, :])
            self.dissimilarity_overlap_container_no_constraint.append(np.vstack((overlap_vector,
                                                                                 dissimilarity_vector)))
            # Set IDOA as 0 for low overlap values
            if np.max(overlap_vector) <= self.min_overlap + self.zero_overlap and not self.percentage:
                self.IDOA_vector[i] = 0
            else:
                filtered_overlap_vector, filtered_dissimilarity_vector = self._filter_od_vectors(
                    overlap_vector, dissimilarity_vector)
                self.dissimilarity_overlap_container.append(np.vstack((filtered_overlap_vector,
                                                                       filtered_dissimilarity_vector)))
                slope = linregress(filtered_overlap_vector, filtered_dissimilarity_vector)[0]  # Calculate the slope
                self.IDOA_vector[i] = slope if not np.isnan(slope) else 0  # If the slope is a valid number set:
                # IDOA = slope
                if np.size(filtered_overlap_vector) < self.min_num_points:
                    self.IDOA_vector[i] = 0
        return self.IDOA_vector

    def calc_idoa_vector(self, second_cohort_ind_dict=(None,)):
        """
        This method calculates the vector of the IDOA values that calculated for a cohort of samples w.r.t the
         reference cohort for all the optional cases(identical or not, single sample or not).
        :param second_cohort_ind_dict: if the cohort is a single sample, this parameter is a tuple of one integer,
                                       if the cohort is not a single sample, this parameter is a dictionary of indexes
                                       in the reference cohort that points to indexes in the cohort.
        :return: IDOA vector.
        """
        if self.cohort.ndim == 1:  # Check if the cohort is a single sample
            return self._calc_idoa_vector_sample_vs_cohort(ind=second_cohort_ind_dict)
        else:
            if self.identical:  # Check if the cohorts considered to be identical
                return self._calc_idoa_vector_cohort_vs_cohort_identical()
            else:
                if second_cohort_ind_dict[0] is not None:  # Check if index dictionary is available
                    return self._calc_idoa_vector_cohort_vs_cohort_custom(ind=second_cohort_ind_dict)
                else:
                    return self._calc_idoa_vector_cohort_vs_cohort_not_identical()