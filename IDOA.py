import numpy as np
from Overlap import Overlap
from Dissimilarity import Dissimilarity
from scipy.stats import linregress

class IDOA:
    """
    This class calculates the IDOA values vector for a cohort or the IDOA value for a sample
    with respect to a reference cohort.
    """

    def __init__(self, ref_cohort, cohort, percentage=50, min_num_points=0, min_overlap=0.5,
                 max_overlap=1, method='percentage'):
        """
        Arguments:
        ref_cohort -- The reference cohort, samples are in the rows.
        cohort -- The cohort, samples are in the rows.
        identical -- If True, both cohorts are considered as identical.
        percentage -- The percentage of the lower overlap points to exclude from IDOA calculation.
        min_num_points -- The minimal number of points to calculate the IDOA.
        min_overlap -- The minimal value of overlap.
        max_overlap -- The maximal value of overlap.
        method -- the method to define the overlap range for IDOA clculation, optional values are: 'percentage' and
                  'min_max'.
        """
        self.ref_cohort = ref_cohort
        if cohort.ndim == 1:
            self.cohort = cohort.reshape(1, -1)
        else:
            self.cohort = cohort
        self.min_num_points = min_num_points
        if type(self.min_num_points) is not int:
            raise ValueError("min_num_points should be a positive integer")
        if self.ref_cohort.ndim != 2:
            raise ValueError("ref_cohort should be a 2D numpy array")
        if self.cohort.ndim != 2:
            raise ValueError("cohort should be a 2D numpy array")
        if self.cohort.shape[1] != self.cohort.shape[1]:
            raise ValueError("The number of columns in cohort and ref_cohort should be equal.")
        try:
            self.min_overlap = min_overlap
            assert isinstance(self.min_overlap, (int, float))
            self.max_overlap = max_overlap
            assert isinstance(self.max_overlap, (int, float))
        except AssertionError:
            raise ValueError("Invalid input values for min_overlap or max_overlap. "
                             "They should be numeric values")
        try:
            assert (0 <= min_overlap < 1)
            assert (min_overlap < max_overlap and (0 < max_overlap <= 1))
        except AssertionError:
            raise ValueError("Invalid input values for min_overlap or max_overlap. "
                             "Their values should be between 0 and 1, with min_overlap less than max_overlap.")
        self.percentage = percentage
        if self.percentage:
            if not (isinstance(self.percentage, int) or isinstance(self.percentage, float)):
                raise ValueError('percentage should be int or float')
            if not (0 <= self.percentage <= 100):
                raise ValueError('percentage must take values between 0 and 100 inclusive.')
        self.method = method
        if not (self.method == 'percentage' or self.method == 'min_max'):
            raise ValueError('Optional values for method are percentage or min_max')
        self.IDOA_vector = np.zeros(self.cohort.shape[0])
        self.d_o_list = []
        self.d_o_list_unconst = []
        self.over_cutoff = []

    def _create_od_vectors(self, sample):
        """
        Arguments:
        sample -- A sample
        Returns:
        overlap_vector and dissimilarity_vector -- this vectors contain the overlap and dissimilarity values
                                                   respectively.
        """
        # Calculate overlap values
        o_objects = [Overlap(ref, sample) for ref in self.ref_cohort if not np.array_equal(ref, sample)]
        overlap_vector = np.array([o_obj.calculate_overlap() for o_obj in o_objects])
        # Calculate dissimilarity values
        d_objects = [Dissimilarity(ref, sample) for ref in self.ref_cohort if not np.array_equal(ref, sample)]
        dissimilarity_vector = np.array([d_obj.calculate_dissimilarity() for d_obj in d_objects])
        return overlap_vector, dissimilarity_vector

    def _filter_od_vectors(self, overlap_vector, dissimilarity_vector):
        """
        Arguments:
        overlap_vector -- Vector that contains the overlap values.
        dissimilarity_vector -- Vector that contains the dissimilarity values.
        Returns:
        filtered_overlap_vector and filtered_dissimilarity_vector -- the original vectors after filtering.
        """
        if self.method == 'percentage':
            overlap_vector_index = np.where(overlap_vector > np.percentile(overlap_vector, self.percentage))
        else:
            overlap_vector_index = np.where(np.logical_and(overlap_vector >= self.min_overlap,
                                                           overlap_vector <= self.max_overlap))
        filtered_overlap_vector = overlap_vector[overlap_vector_index]
        filtered_dissimilarity_vector = dissimilarity_vector[overlap_vector_index]
        self.over_cutoff.append(filtered_overlap_vector.size)
        return filtered_overlap_vector, filtered_dissimilarity_vector

    def calc_idoa_vector(self):
        """
        This is a private method that calculates the IDOA vector for a cohort w.r.t to itself.

        Returns:
        IDOA vector -- A vector that contains the IDOA values for each sample in the cohort. If the cohort contains
                       only one sample, the IDOA vector is squeezed to a scalar.
        """
        for i, smp in enumerate(self.cohort):
            overlap_vector, dissimilarity_vector = self._create_od_vectors(smp)
            self.d_o_list_unconst.append(np.vstack((overlap_vector, dissimilarity_vector)))
            # Set IDOA as 0 for low overlap values
            if np.max(overlap_vector) <= self.min_overlap and not self.percentage:
                self.IDOA_vector[i] = 0
            else:
                filtered_overlap_vector, filtered_dissimilarity_vector = self._filter_od_vectors(overlap_vector,
                                                                                                 dissimilarity_vector)
                self.d_o_list.append(np.vstack((filtered_overlap_vector, filtered_dissimilarity_vector)))
                if filtered_overlap_vector.size == 0:
                    self.IDOA_vector[i] = 0
                else:
                    slope = linregress(filtered_overlap_vector, filtered_dissimilarity_vector)[0]  # Calculate the slope
                    self.IDOA_vector[i] = slope if not np.isnan(slope) else 0  # If the slope is a valid number set:
                    # IDOA = slope
                    if np.size(filtered_overlap_vector) < self.min_num_points:
                        self.IDOA_vector[i] = 0
        return np.squeeze(self.IDOA_vector)

    def additional_info(self):
        """
        Returns:
        info_dict -- A dictionary that contains the following information:
                     d_o_list -- A list that contains the overlap and dissimilarity values for each sample
                                 in the cohort.
                     d_o_list_unconst -- A list that contains the overlap and dissimilarity values for each
                                         sample without any constraints on the overlap values.
                     over_cutoff -- A list that contains the number of points used for IDOA calculation for each
                                    sample in the cohort.
        """
        info_dict = {'d_o_list': self.d_o_list,
                     'd_o_list_unconst': self.d_o_list_unconst,
                     'over_cutoff': np.squeeze(self.over_cutoff)}
        return info_dict