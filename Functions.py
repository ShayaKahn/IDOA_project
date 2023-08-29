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