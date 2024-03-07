import numpy as np
from IDOA import IDOA
from scipy.spatial.distance import braycurtis
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.metrics import pairwise_distances

def calc_initial_conditions(number_of_species, number_of_samples, min_val=0.6, max_val=0.9):
    prob_vector = np.random.uniform(min_val, max_val, (number_of_samples, number_of_species))
    y0 = np.zeros((number_of_samples, number_of_species))
    rand_vector = np.random.rand(number_of_samples, number_of_species)
    mask = rand_vector < prob_vector
    y0[mask] = np.random.rand(np.sum(mask))
    return np.squeeze(y0)

def create_interaction_matrix(num_species, delta_int=0.01, p=0.25):
    # Create a num_species x num_species matrix filled with zeros
    interaction_matrix = np.zeros((num_species, num_species))
    # Generate random numbers between -delta and delta
    random_numbers = np.random.uniform(-delta_int, delta_int, size=(num_species, num_species))
    # Create a mask based on the probability p
    mask = np.random.rand(num_species, num_species) < p
    # Apply the mask to the random numbers
    interaction_matrix[mask] = random_numbers[mask]
    return interaction_matrix

def normalize_cohort(cohort):
    # normalization function
    if cohort.ndim == 1:
        cohort_normalized = cohort / cohort.sum()
    else:
        cohort_normalized = np.divide(cohort, np.linalg.norm(cohort, ord=1, axis=1, keepdims=True))
    return cohort_normalized

def find_cutoff(x, y, num_neighbors):
    # Calculate moving averages
    moving_averages = [np.mean(y[i - num_neighbors: i + num_neighbors + 1]
                               ) for i in range(num_neighbors, len(x) - num_neighbors)]

    index = None

    # Iterate over the array in reverse order and find the index where the average starts to decrease
    for i in range(len(moving_averages) - 1, 0, -1):
        if moving_averages[i] > moving_averages[i - 1]:
            index = i + num_neighbors
            break

    if index is None:
        raise ValueError("No cutoff found")

    return x[index]

def remove_zeros(sample_first_idoa, sample_sec_idoa):
    """
    Remove the zero values from the IDOA vectors
    :param sample_first_idoa: IDOA values for the first sample
    :param sample_sec_idoa: IDOA values for the second sample
    :return: IDOA vectors without the zero values in both idoa vectors
    """

    samples_mat = np.vstack((sample_first_idoa, sample_sec_idoa))
    ind = np.where(samples_mat.sum(axis=0) != 0)[0]
    return sample_first_idoa[ind], sample_sec_idoa[ind]

def calculate_confusion_matrix(control_control, asd_control, asd_asd,
                               control_asd):
    """
    Calculate the confusion matrix for the given data
    :param control_control: IDOA values for the control samples w.r.t. the control samples
    :param asd_control: IDOA values for the control samples w.r.t. the ASD samples
    :param asd_asd: IDOA values for the ASD samples w.r.t. the ASD samples
    :param control_asd: IDOA values for the ASD samples w.r.t. the control samples
    :return: confusion matrix
    """
    control_control_new, asd_control_new = remove_zeros(control_control, asd_control)
    asd_asd_new, control_asd_new = remove_zeros(asd_asd, control_asd)

    m = len(control_control_new)
    n = len(asd_asd_new)

    healthy_count = np.sum(control_control_new < asd_control_new)
    asd_count = np.sum(asd_asd_new < control_asd_new)

    # True Positives: ASD subjects correctly classified as ASD
    tp = asd_count
    # False Negatives: ASD subjects incorrectly classified as Healthy
    fn = n - asd_count
    # False Positives: Healthy subjects incorrectly classified as ASD
    fp = m - healthy_count
    # True Negatives: Healthy subjects correctly classified as Healthy
    tn = healthy_count

    confusion_matrix = np.array([[tp, fp], [fn, tn]])

    return confusion_matrix

def construct_threshold_matrix(delta_vector, num_rows, epsilon):
    """
    Constructs a threshold matrix for the ROC curve
    :param delta_vector: Vector with the differences between the measured values
    :param num_rows: Number of rows for the threshold matrix
    :param epsilon: Epsilon value for the threshold matrix
    return: Threshold matrix
    """
    max_delta = max(delta_vector)
    min_delta = min(delta_vector)

    # Create a matrix with num_rows rows and columns based on delta_vector
    threshold_matrix = np.tile(delta_vector, (num_rows, 1))

    # Create a linspace for the additional values
    additional_values = np.linspace(-max_delta - epsilon,
                                    -min_delta + epsilon, num_rows).reshape(-1, 1)

    # Add the additional values to each row of the matrix
    threshold_matrix = threshold_matrix + additional_values

    return threshold_matrix

def calc_bray_curtis_dissimilarity(first_cohort, second_cohort):
    mean_dist_vector = []
    for smp in second_cohort:
        # Find the indices of rows that are equal to the specific vector
        indices_to_remove = np.all(first_cohort == smp, axis=1)
        # Remove the rows using boolean indexing
        filtered_first_cohort = first_cohort[~indices_to_remove]
        mean_bc = pairwise_distances(smp.reshape(1, -1), filtered_first_cohort, metric='braycurtis').mean()
        mean_dist_vector.append(mean_bc)
    return np.array(mean_dist_vector)



