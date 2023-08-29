import numpy as np
from scipy.sparse import lil_matrix
import random

class ShuffledCohort:
    def __init__(self, cohort):
        """
        :param cohort: a numpy matrix with samples in rows and species in columns
        """
        self.cohort = cohort
        if not isinstance(self.cohort, np.ndarray) or self.cohort.ndim != 2:
            print("Error: The provided cohort is not a 2-dimensional numpy matrix.")
        try:
            nonzero_counts = np.count_nonzero(self.cohort, axis=0)
            assert np.all(nonzero_counts >= 1)
        except AssertionError:
            print("Error: The condition of all columns with more than 1 non-zero value is not satisfied.")
        self.shuffled_cohort = lil_matrix(cohort.T)

    def create_shuffled_cohort(self):
        for j, abundances in enumerate(self.shuffled_cohort.data):
            self.shuffled_cohort.data[j] = self._derange_list(abundances)
        return self._normalize_cohort(self.shuffled_cohort.toarray().T)

    @ staticmethod
    def _is_derangement(original, shuffled):
        return all(o != s for o, s in zip(original, shuffled))

    def _derange_list(self, lst):
        if len(lst) == 1:
            return lst

        shuffled = lst.copy()
        while not self._is_derangement(lst, shuffled):
            random.shuffle(shuffled)
        return shuffled

    @staticmethod
    def _normalize_cohort(cohort):
        # Normalize cohort to sum to 1
        if cohort.ndim == 1:
            cohort_normalized = cohort / cohort.sum()
        else:
            cohort_normalized = cohort / np.linalg.norm(cohort, ord=1, axis=1, keepdims=True)
        return cohort_normalized
