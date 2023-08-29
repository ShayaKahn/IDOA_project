import numpy as np
from Overlap import Overlap
from Dissimilarity import Dissimilarity

class DOC:
    """
    This class calculates the DOC matrix of a given cohort.
    """
    def __init__(self, cohort, norm=True):
        """
        param cohort: a matrix, samples are in the rows.
        """
        self.cohort = cohort
        if self.cohort.ndim != 2:
            raise ValueError("ref_cohort should be a 2D numpy array")
        self.num_samples = cohort.shape[0]
        self.norm = norm

    def calc_doc(self):
        """
        :return: matrix, the first row is the dissimilarity values for all the sample pairs of the cohort,
                 the second row is for the overlap.
        """
        num_samples = self.num_samples

        # Calculate overlap values
        o = np.array([Overlap(self.cohort[j, :], self.cohort[i, :], norm=self.norm).calculate_overlap()
                      for j in range(num_samples - 1) for i in range(j + 1, num_samples)])

        # Calculate dissimilarity values
        d = np.array([Dissimilarity(self.cohort[j, :], self.cohort[i, :]).calculate_dissimilarity()
                      for j in range(num_samples - 1) for i in range(j + 1, num_samples)])

        # Combine overlap and dissimilarity into a matrix
        doc_mat = np.vstack((o, d))

        return doc_mat

    def bootstrap(self):
        """
        This method applies bootstrap procedure, using leave one out,
        it calculates the DOC matrix for all the samples
        excluding single sample each time.
        :return: a container with all the DOC matrices for the procedure.
        """
        bootstrap_mat_container = []
        for i in range(0, self.num_samples):
            resampled_cohort = np.delete(self.cohort, i, axis=0)  # Remove the ith sample
            resampled_doc = DOC(resampled_cohort)  # Initiate DOC
            bootstrap_mat_container.append(resampled_doc.calc_doc())  # Apply DOC
        return bootstrap_mat_container
