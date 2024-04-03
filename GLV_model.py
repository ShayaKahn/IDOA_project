import numpy as np
from scipy.integrate import solve_ivp
from GLV_functions import f, event

class Glv:
    """
    This class is responsible to solve the GLV model with verification of reaching the steady state
    for a given parameters.
    """
    def __init__(self, n_samples, n_species, delta, r, s,
                 interaction_matrix, initial_cond, final_time, max_step):
        """
        :param n_samples: The number of samples you are need to compute.
        :param n_species: The number of species at each sample.
        :param delta: This parameter is responsible for the stop condition at the steady state.
        :param r: growth rate vector of shape (,n_species).
        :param s: logistic growth term vector of size (,n_species).
        :param interaction_matrix: interaction matrix of shape (n_species, n_species).
        :param initial_cond: set of initial conditions for each sample. If n_samples=1, the shape is (,n_species).
        If n_samples=m for m!=1 so the shape is (n_samples, n_species)
        :param final_time: the final time of the integration.
        :param max_step: maximal allowed step size.
        """
        # Check if n_samples and n_species are integers greater than 0
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be an integer greater than 0.")
        if not isinstance(n_species, int) or n_species <= 0:
            raise ValueError("n_species must be an integer greater than 0.")

        # Check if delta is a number between 0 and 1
        if not (0 < delta < 1):
            raise ValueError("delta must be a number between 0 and 1.")

        # Check if r and s are numpy vectors of length n_species
        if not (isinstance(r, np.ndarray) and r.shape == (n_species,)):
            raise ValueError("r must be a numpy vector of length n_species.")
        if not (isinstance(s, np.ndarray) and s.shape == (n_species,)):
            raise ValueError("s must be a numpy vector of length n_species.")

        # Check if interaction_matrix is a numpy matrix of shape (n_species, n_species)
        if not (isinstance(interaction_matrix, np.ndarray) and interaction_matrix.shape == (n_species, n_species)):
            raise ValueError("interaction_matrix must be a numpy matrix of shape (n_species, n_species).")

        # Check if initial_cond is a numpy matrix of shape (n_species, n_samples) with non-negative values
        if not (isinstance(initial_cond, np.ndarray) and initial_cond.shape == (n_samples, n_species) and np.all(
                initial_cond >= 0)):
            raise ValueError(
                "initial_cond must be a numpy matrix of shape (n_samples, n_species) with non-negative values.")

        # Check if final_time is a number greater than zero
        if not (isinstance(final_time, (int, float)) and final_time > 0):
            raise ValueError("final_time must be a number greater than zero.")

        # Check if max_step is a number greater than zero and smaller than final_time
        if not (isinstance(max_step, (int, float)) and 0 < max_step < final_time):
            raise ValueError("max_step must be a number greater than zero and smaller than final_time.")

        self.smp = n_samples
        self.n = n_species
        self.delta = delta
        self.r = r
        self.s = s
        self.A = interaction_matrix
        self.Y = initial_cond
        self.final_time = final_time
        self.max_step = max_step

        # Initiation.
        self.Final_abundances = np.zeros((self.n, self.smp))
        self.Final_abundances_single_sample = np.zeros(self.n)

    def solve(self):
        """
        This function updates the final abundances, rows are the species and columns represent the samples.
        """

        # Set the parameters to the functions f and event.
        f_with_params = lambda t, x: f(t, x, self.r, self.s, self.A, self.delta)
        event_with_params = lambda t, x: event(t, x, self.r, self.s, self.A, self.delta)

        # event definitions
        event_with_params.terminal = True
        event_with_params.direction = -1

        if self.smp > 1:  # Solution for cohort.
            for m in range(self.smp):
                print(m)
                # solve GLV up to time span.
                sol = solve_ivp(f_with_params, (0, self.final_time), self.Y[m, :], max_step=self.max_step,
                                events=event_with_params)

                if np.size(sol.t_events[0]) == 1:
                    self.Final_abundances[:, m] = sol.y[:, -1]
                else:
                    raise RuntimeError("The expected event did not occur during the integration.")

            final_abundances = self.Final_abundances
            return self.normalize_cohort(final_abundances.T)

        else:  # Solution for single sample.
            sol = solve_ivp(f, (0, self.final_time),
                            self.Y[:], max_step=self.max_step, events=event)

            if np.size(sol.t_events[0]) == 1:
                self.Final_abundances_single_sample[:] = sol.y[:, -1]
            else:
                raise RuntimeError("The expected event did not occur during the integration.")

        final_abundances = self.Final_abundances_single_sample
        return self.normalize_cohort(final_abundances)

    @staticmethod
    def normalize_cohort(cohort):
        # normalization function
        if cohort.ndim == 1:
            cohort_normalized = cohort / cohort.sum()
        else:
            cohort_normalized = cohort / np.linalg.norm(cohort, ord=1, axis=1, keepdims=True)
        return cohort_normalized