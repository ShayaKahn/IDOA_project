from unittest import TestCase
from IDOA import IDOA
import numpy as np
from GLV_model import Glv
from Overlap import Overlap
from Shuffle_cohort import ShuffledCohort

class Test_GLV(TestCase):
    def setUp(self) -> None:
        s = np.ones(100)
        r = np.random.uniform(0, 1, 100)
        time_span = 200
        max_step = 0.5
        delta = 0.00001

        def calc_matrix(num_of_species):
            interaction_matrix = np.zeros([num_of_species, num_of_species])
            p = 0.25
            for row, col in np.ndindex(interaction_matrix.shape):
                if np.random.uniform(0, 1) < p:
                    interaction_matrix[row, col] = np.random.uniform(-0.05, 0.05)
                else:
                    interaction_matrix[row, col] = 0
            return interaction_matrix

        A = calc_matrix(100)

        def calc_initial_condition(number_of_species):
            prob_vector = np.random.uniform(0.6, 0.9, number_of_species)
            y0 = np.zeros(number_of_species)
            for i in range(0, number_of_species):
                if np.random.uniform(0, 1) < prob_vector[i]:
                    y0[i] = np.random.uniform(0, 1)
                else:
                    y0[i] = 0
            return y0

        def clac_set_of_initial_conditions(num_species, num_samples):
            init_cond_set = np.zeros([num_samples, num_species])
            for i in range(0, num_samples):
                init_cond_set[:][i] = calc_initial_condition(num_species)
            return init_cond_set

        Y0 = clac_set_of_initial_conditions(100, 10)

        self.glv = Glv(n_samples=10, n_species=100, delta=delta, r=r, s=s, interaction_matrix=A, initial_cond=Y0,
                       final_time=time_span, max_step=max_step)

    def test_solve(self):
        final_abundances = self.glv.solve()
        print(final_abundances)

class Test_Overlap(TestCase):
    def setUp(self) -> None:
        self.first_sample = np.array([0.1, 0, 0.2, 0.4, 0, 0, 0.1, 0.2])
        self.second_sample = np.array([0.1, 0, 0.2, 0.4, 0.1, 0, 0, 0.2])

    def test_calculate_overlap(self):
        overlap = Overlap(self.first_sample, self.second_sample, overlap_type='Jaccard').calculate_overlap()

class TestIDOA(TestCase):
    """
    This class tests the IDOA class.
    """
    def setUp(self) -> None:
        self.ref_cohort = np.array([[1, 3, 0, 7, 14, 0, 0.5, 8, 44, 0.01],
                                    [11, 0, 1, 13, 0, 5, 0.1, 8.5, 4, 0],
                                    [2, 2, 1, 0, 10, 0, 0, 0, 4, 0.001],
                                    [30, 0, 0, 1, 0, 0, 0.18, 5, 4, 0],
                                    [30, 0, 0, 1, 0, 0, 0.18, 5, 4, 0],
                                    [13, 1, 1, 1, 3, 4, 0.1, 15, 1, 9],
                                    [8, 2, 4, 5, 6, 8, 0, 5, 3, 3],
                                    [3, 1, 3, 1, 4, 7, 2, 50, 3, 1]])

        self.single_sample_included = np.array([1, 3, 0, 7, 14, 0, 0.5, 8, 44, 0.01])

        self.single_sample_not_included = np.array([4, 0, 0, 18, 1, 0, 0, 2, 0, 80])

        self.cohort_included = np.array([[1, 3, 0, 7, 14, 0, 0.5, 8, 44, 0.01],
                                         [1, 10, 20, 0, 4, 8, 0.01, 19, 0, 0]])

        self.cohort_not_included = np.array([[2, 13, 1, 0, 0, 0, 0, 3, 0, 0.1],
                                             [1, 10, 20, 0, 4, 8, 0.01, 19, 0, 0]])

        self.idoa_ss_included = IDOA(self.ref_cohort, self.single_sample_included, min_overlap=0.2, max_overlap=1,
                                     zero_overlap=0, identical=False)
        self.idoa_ss_included_vector = self.idoa_ss_included.calc_idoa_vector(second_cohort_ind_dict=(0,))

        self.idoa_ss_not_included = IDOA(self.ref_cohort, self.single_sample_not_included, min_overlap=0.2,
                                         max_overlap=1, zero_overlap=0, identical=False)
        self.idoa_ss_not_included_vector = self.idoa_ss_not_included.calc_idoa_vector()

        self.idoa_cohort_included = IDOA(self.ref_cohort, self.cohort_included, min_overlap=0.2,
                                         max_overlap=1, zero_overlap=0, identical=False)
        self.idoa_cohort_included_vector = self.idoa_cohort_included.calc_idoa_vector(second_cohort_ind_dict={0: (0,)})

        self.idoa_cohort_not_included = IDOA(self.ref_cohort, self.cohort_not_included, min_overlap=0.2,
                                             max_overlap=1, zero_overlap=0, identical=False)
        self.idoa_cohort_not_included_vector = self.idoa_cohort_not_included.calc_idoa_vector()

        self.idoa_identical = IDOA(self.ref_cohort, self.ref_cohort, min_overlap=0.2,
                                   max_overlap=1, zero_overlap=0, identical=True)
        self.idoa_identical_vector = self.idoa_identical.calc_idoa_vector()

    def test_calc_idoa_vector(self):

        # Test single sample included
        self.assertEqual(np.size(self.idoa_ss_included_vector), 1)
        self.assertEqual(np.size(self.idoa_ss_included.dissimilarity_overlap_container_no_constraint[0][0]),
                         np.size(self.ref_cohort, axis=0) - 1)
        self.assertFalse(np.any(self.idoa_ss_included.dissimilarity_overlap_container_no_constraint[0][0] == 1),
                         "Overlap contains 1")
        self.assertFalse(np.any(self.idoa_ss_included.dissimilarity_overlap_container_no_constraint[0][1] == 0),
                         "Dissimilarity contains 0")

        # Test single sample not included
        self.assertEqual(np.size(self.idoa_ss_not_included_vector), 1)
        self.assertEqual(np.size(self.idoa_ss_not_included.dissimilarity_overlap_container_no_constraint[0][0]),
                         np.size(self.ref_cohort, axis=0))

        # Test cohort included
        self.assertEqual(np.size(self.idoa_cohort_included_vector), np.size(self.cohort_included, axis=0))
        self.assertEqual(np.size(self.idoa_cohort_included.dissimilarity_overlap_container_no_constraint[0][0]),
                         np.size(self.ref_cohort, axis=0) - 1)
        self.assertFalse(np.any(self.idoa_cohort_included.dissimilarity_overlap_container_no_constraint[0][0] == 1),
                         "Overlap contains 1")
        self.assertFalse(np.any(self.idoa_cohort_included.dissimilarity_overlap_container_no_constraint[0][1] == 0),
                         "Dissimilarity contains 0")

        # Test cohort not included
        self.assertEqual(np.size(self.idoa_cohort_not_included_vector),
                         np.size(self.idoa_cohort_not_included_vector, axis=0))
        self.assertEqual(np.size(self.idoa_cohort_not_included.dissimilarity_overlap_container_no_constraint[0][0]),
                         np.size(self.ref_cohort, axis=0))

        # Test identical
        self.assertEqual(np.size(self.idoa_identical_vector), np.size(self.ref_cohort, axis=0))
        self.assertEqual(np.size(self.idoa_identical.dissimilarity_overlap_container_no_constraint[0][0]),
                         np.size(self.ref_cohort, axis=0) - 1)
        self.assertFalse(np.any(self.idoa_identical.dissimilarity_overlap_container_no_constraint[0][0] == 1),
                         "Overlap contains 1")
        self.assertFalse(np.any(self.idoa_identical.dissimilarity_overlap_container_no_constraint[0][1] == 0),
                         "Dissimilarity contains 0")

class TestShuffledCohort(TestCase):
    def setUp(self) -> None:
        self.cohort = np.array([[0.1, 0.4, 0.3, 0, 0.2], [0, 0.2, 0.4, 0.4, 0], [0.3, 0.3, 0.1, 0.2, 0.1],
                                [0, 0, 1, 0, 0]])

        #print(self.cohort)
        self.shuffled_cohort_object = ShuffledCohort(self.cohort)
        self.shuffled_cohort = self.shuffled_cohort_object.create_shuffled_cohort()

    def test_shuffle_cohort(self):

        mask_input = (self.cohort == 0)
        mask_output = (self.shuffled_cohort == 0)

        # Check if the masks are the same, i.e., zeros are in the same places
        self.assertTrue(np.array_equal(mask_input, mask_output))
