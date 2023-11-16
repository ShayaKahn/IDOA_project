from Overlap import Overlap
from Dissimilarity import Dissimilarity
import numpy as np

def normalize_data(data):
    """
    :param data: Matrix of data
    :return: Normalized matrix of the data.
    """
    norm_factors = np.sum(data, axis=0)
    norm_data = np.array([data[:, i] / norm_factors[i] for i in range(0, np.size(norm_factors))])
    return norm_data.T

def idoa(sample, cohort, overlap_vector, dissimilarity_vector):
    """
    :param sample: single sample
    :param cohort: cohort that consists of m samples
    :param overlap_vector: empty vector size m
    :param dissimilarity_vector: empty vector size m
    :return: overlap and dissimilarity vectors for larger than 0.5 overlap values
    """
    for i in range(0, np.size(cohort, axis=1)):
        o = Overlap(sample, cohort[:, i])
        d = Dissimilarity(sample, cohort[:, i])
        overlap_vector[i] = o.calculate_overlap()
        dissimilarity_vector[i] = d.calculate_dissimilarity()
    # Indexes of the overlap vector that are greater than 0.5.
    overlap_vector_index = np.where(np.logical_and(overlap_vector >= 0.5, overlap_vector <= 1))
    new_overlap_vector = overlap_vector[overlap_vector_index]
    new_dissimilarity_vector = dissimilarity_vector[overlap_vector_index]
    return new_overlap_vector, new_dissimilarity_vector

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

delta_interacrions = 0.01
def calc_matrix(num_of_species):
    interaction_matrix = np.zeros([num_of_species, num_of_species])
    p = 0.25
    for row, col in np.ndindex(interaction_matrix.shape):
        if np.random.uniform(0, 1) < p:
            interaction_matrix[row, col] = np.random.uniform(-delta_interacrions, delta_interacrions)
        else:
            interaction_matrix[row, col] = 0
    return interaction_matrix

def normalize_cohort(cohort):
    # normalization function
    if cohort.ndim == 1:
        cohort_normalized = cohort / cohort.sum()
    else:
        cohort_normalized = cohort / np.linalg.norm(cohort, ord=1, axis=1, keepdims=True)
    return cohort_normalized