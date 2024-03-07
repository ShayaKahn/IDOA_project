import numpy as np
from IDOA import IDOA
from Functions import calc_bray_curtis_dissimilarity
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.metrics import pairwise_distances

class Classifier:
    def __init__(self, first_data, sec_data, num_iters, method, test_method='KS'):
        self.first_data = first_data
        self.sec_data = sec_data
        self.num_iters = num_iters
        self.method = method
        self.test_method = test_method

    @ staticmethod
    def _bootstrapping(data, num_iters):
        size = data.shape[0]
        resampled_data_ind_mat = np.random.choice(np.arange(size), size=(num_iters, size), replace=True)
        return resampled_data_ind_mat

    @staticmethod
    def _conf_mat_uncertain(pred_labels_first, pred_labels_second,
                            true_labels_first, true_labels_second,
                            p_vals_first, p_vals_second, alpha=0.05):

        tp_mask = (pred_labels_first == true_labels_first) & (p_vals_first < alpha)
        tp = np.sum(tp_mask)
        fn_mask = (pred_labels_first != true_labels_first) & (p_vals_first < alpha)
        fn = np.sum(fn_mask)
        tn_mask = (pred_labels_second == true_labels_second) & (p_vals_second < alpha)
        tn = np.sum(tn_mask)
        fp_mask = (pred_labels_second != true_labels_second) & (p_vals_second < alpha)
        fp = np.sum(fp_mask)
        uncertain_first = np.where(p_vals_first > alpha)[0].shape[0]
        uncertain_second = np.where(p_vals_second > alpha)[0].shape[0]
        confusion_matrix = np.array([[tp, fp], [fn, tn], [uncertain_first, uncertain_second]])

        return confusion_matrix

    def _calc_p_value_IDOA(self, min_num_points, min_overlap, method, percentage, max_overlap):
        p_vals_vector = []
        resampled_second_ind = self._bootstrapping(self.sec_data, self.num_iters)
        resampled_sec = [self.sec_data[ind, :] for ind in resampled_second_ind]
        for i, sample in enumerate(self.first_data):
            resampled_first_ind = self._bootstrapping(np.delete(self.first_data, i, axis=0), self.num_iters)
            resampled_first = [self.first_data[ind, :] for ind in resampled_first_ind]
            idoa_vals_first = [IDOA(cohort, sample, percentage, min_num_points, min_overlap, max_overlap, method
                                    ).calc_idoa_vector() for cohort in resampled_first]
            idoa_vals_sec = [IDOA(cohort, sample, percentage, min_num_points, min_overlap, max_overlap, method
                                  ).calc_idoa_vector() for cohort in resampled_sec]
            if self.test_method == 'KS':
                _, p_val = ks_2samp(idoa_vals_first, idoa_vals_sec)
                p_vals_vector.append(p_val)
            elif self.test_method == 'MW':
                _, p_val = mannwhitneyu(idoa_vals_first, idoa_vals_sec)
                p_vals_vector.append(p_val)

        resampled_first_ind = self._bootstrapping(self.first_data, self.num_iters)
        resampled_first = [self.first_data[ind, :] for ind in resampled_first_ind]
        for i, sample in enumerate(self.sec_data):
            resampled_second_ind = self._bootstrapping(np.delete(self.sec_data, i, axis=0), self.num_iters)
            resampled_sec = [self.sec_data[ind, :] for ind in resampled_second_ind]
            idoa_vals_first = [IDOA(cohort, sample, percentage, min_num_points, min_overlap, max_overlap, method
                                    ).calc_idoa_vector() for cohort in resampled_first]
            idoa_vals_sec = [IDOA(cohort, sample, percentage, min_num_points, min_overlap, max_overlap, method
                                  ).calc_idoa_vector() for cohort in resampled_sec]
            if self.test_method == 'KS':
                _, p_val = ks_2samp(idoa_vals_first, idoa_vals_sec)
                p_vals_vector.append(p_val)
            elif self.test_method == 'MW':
                _, p_val = mannwhitneyu(idoa_vals_first, idoa_vals_sec)
                p_vals_vector.append(p_val)
        return np.array(p_vals_vector)

    def _calc_p_value_BC(self):
        p_vals_vector = []
        resampled_second_ind = self._bootstrapping(self.sec_data, self.num_iters)
        resampled_sec = [self.sec_data[ind, :] for ind in resampled_second_ind]
        for i, sample in enumerate(self.first_data):
            resampled_first_ind = self._bootstrapping(np.delete(self.first_data, i, axis=0), self.num_iters)
            resampled_first = [self.first_data[ind, :] for ind in resampled_first_ind]
            bc_vals_first = []
            for cohort in resampled_first:
                # Find the indices of rows that are equal to the specific vector
                indices_to_remove = np.all(cohort == sample, axis=1)
                # Remove the rows using boolean indexing
                filtered_cohort = cohort[~indices_to_remove]
                mean_bc = pairwise_distances(sample.reshape(1, -1), filtered_cohort, metric='braycurtis').mean()
                bc_vals_first.append(mean_bc)
            bc_vals_sec = []
            for cohort in resampled_sec:
                # Find the indices of rows that are equal to the specific vector
                indices_to_remove = np.all(cohort == sample, axis=1)
                # Remove the rows using boolean indexing
                filtered_cohort = cohort[~indices_to_remove]
                mean_bc = pairwise_distances(sample.reshape(1, -1), filtered_cohort, metric='braycurtis').mean()
                bc_vals_sec.append(mean_bc)
            if self.test_method == 'KS':
                _, p_val = ks_2samp(bc_vals_first, bc_vals_sec)
                p_vals_vector.append(p_val)
            elif self.test_method == 'MW':
                _, p_val = mannwhitneyu(bc_vals_first, bc_vals_sec)
                p_vals_vector.append(p_val)

        resampled_first_ind = self._bootstrapping(self.first_data, self.num_iters)
        resampled_first = [self.first_data[ind, :] for ind in resampled_first_ind]
        for i, sample in enumerate(self.sec_data):
            resampled_second_ind = self._bootstrapping(np.delete(self.sec_data, i, axis=0), self.num_iters)
            resampled_sec = [self.sec_data[ind, :] for ind in resampled_second_ind]
            bc_vals_first = []
            for cohort in resampled_first:
                # Find the indices of rows that are equal to the specific vector
                indices_to_remove = np.all(cohort == sample, axis=1)
                # Remove the rows using boolean indexing
                filtered_cohort = cohort[~indices_to_remove]
                mean_bc = pairwise_distances(sample.reshape(1, -1), filtered_cohort, metric='braycurtis').mean()
                bc_vals_first.append(mean_bc)
            bc_vals_sec = []
            for cohort in resampled_sec:
                # Find the indices of rows that are equal to the specific vector
                indices_to_remove = np.all(cohort == sample, axis=1)
                # Remove the rows using boolean indexing
                filtered_cohort = cohort[~indices_to_remove]
                mean_bc = pairwise_distances(sample.reshape(1, -1), filtered_cohort, metric='braycurtis').mean()
                bc_vals_sec.append(mean_bc)
            if self.test_method == 'KS':
                _, p_val = ks_2samp(bc_vals_first, bc_vals_sec)
                p_vals_vector.append(p_val)
            elif self.test_method == 'MW':
                _, p_val = mannwhitneyu(bc_vals_first, bc_vals_sec)
                p_vals_vector.append(p_val)
        return np.array(p_vals_vector)

    def _calc_p_value(self, min_num_points=10, min_overlap=0.5, method='min_max', percentage=50, max_overlap=1):
        if self.method == 'IDOA':
            return self._calc_p_value_IDOA(min_num_points, min_overlap, method, percentage, max_overlap)

        elif self.method == 'braycurtis':
            return self._calc_p_value_BC()

    def _classify_IDOA(self, true_labels_first, true_labels_second, alpha, min_num_points, min_overlap,
                       method, percentage, max_overlap):
        idoa_obj_first_first = IDOA(self.first_data, self.first_data, percentage, min_num_points, min_overlap,
                                    max_overlap, method)
        idoa_first_first = idoa_obj_first_first.calc_idoa_vector()
        idoa_obj_sec_first = IDOA(self.sec_data, self.first_data, percentage, min_num_points, min_overlap, max_overlap,
                                  method)
        idoa_sec_first = idoa_obj_sec_first.calc_idoa_vector()
        pred_labels_first = np.array([1 if x < y else 0 for x, y in zip(idoa_first_first, idoa_sec_first)])
        idoa_obj_sec_sec = IDOA(self.sec_data, self.sec_data, percentage, min_num_points, min_overlap, max_overlap,
                                method)
        idoa_sec_sec = idoa_obj_sec_sec.calc_idoa_vector()
        idoa_obj_first_sec = IDOA(self.first_data, self.sec_data, percentage, min_num_points, min_overlap, max_overlap,
                                  method)
        idoa_first_sec = idoa_obj_first_sec.calc_idoa_vector()
        pred_labels_sec = np.array([0 if x < y else 1 for x, y in zip(idoa_sec_sec, idoa_first_sec)])
        pred_labels = np.concatenate((pred_labels_first, pred_labels_sec))
        p_values = self._calc_p_value(min_num_points, min_overlap, method, percentage, max_overlap)
        conf_mat = self._conf_mat_uncertain(pred_labels_first, pred_labels_sec,
                                            true_labels_first, true_labels_second,
                                            p_values[0:self.first_data.shape[0]],
                                            p_values[self.first_data.shape[0]:], alpha=alpha)
        info = {'pred labels': pred_labels, 'p values': p_values, 'conf mat': conf_mat}
        return info

    def _classify_BC(self, true_labels_first, true_labels_second, alpha):
        mean_bc_first_first = calc_bray_curtis_dissimilarity(self.first_data, self.first_data)
        mean_bc_sec_first = calc_bray_curtis_dissimilarity(self.sec_data, self.first_data)
        mean_bc_sec_sec = calc_bray_curtis_dissimilarity(self.sec_data, self.sec_data)
        mean_bc_first_sec = calc_bray_curtis_dissimilarity(self.first_data, self.sec_data)
        pred_labels_first = np.array([1 if x < y else 0 for x, y in zip(mean_bc_first_first, mean_bc_sec_first)])
        pred_labels_sec = np.array([0 if x < y else 1 for x, y in zip(mean_bc_sec_sec, mean_bc_first_sec)])
        pred_labels = np.concatenate((pred_labels_first, pred_labels_sec))
        p_values = self._calc_p_value()
        conf_mat = self._conf_mat_uncertain(pred_labels_first, pred_labels_sec,
                                            true_labels_first, true_labels_second,
                                            p_values[0:self.first_data.shape[0]],
                                            p_values[self.first_data.shape[0]:], alpha=alpha)
        info = {'pred labels': pred_labels, 'p values': p_values, 'conf mat': conf_mat}
        return info

    def classify(self, alpha=0.05, min_num_points=10, min_overlap=0.5, method='min_max', percentage=50, max_overlap=1):
        true_labels_first = np.ones(self.first_data.shape[0])
        true_labels_second = np.zeros(self.sec_data.shape[0])
        if self.method == 'IDOA':
            return self._classify_IDOA(true_labels_first, true_labels_second, alpha, min_num_points, min_overlap,
                                  method, percentage, max_overlap)
        elif self.method == 'braycurtis':
            return self._classify_BC(true_labels_first, true_labels_second, alpha)


    """
    def __init__(self, first_data, sec_data):
        self.first_data = first_data
        self.sec_data = sec_data
        self.first_data_labels = np.ones(self.first_data.shape[0])
        self.sec_data_labels = np.zeros(self.sec_data.shape[0])

    def classify(self, method='IDOA', uncertainty_method='std'):
        if method == 'IDOA':
            delta_IDOA_first, delta_IDOA_sec, pred_labels_first, pred_labels_sec = self._idoa_pred_delta()
            delta_tot = np.concatenate((delta_IDOA_first, delta_IDOA_sec))
            unclass_detect_first, unclass_detect_sec = self._cals_unclassified(delta_IDOA_first,
                                                                               delta_IDOA_sec, uncertainty_method)
            conf_mat = self._create_confusion_matrix(pred_labels_first, pred_labels_sec, unclass_detect_first,
                                                     unclass_detect_sec)
            return conf_mat


        elif method == 'bray_curtis':
            delta_bc_first, delta_bc_sec, pred_labels_first, pred_labels_sec = self._bc_pred_delta()
            unclass_detect_first, unclass_detect_sec = self._cals_unclassified(delta_bc_first,
                                                                               delta_bc_sec, uncertainty_method)

            conf_mat = self._create_confusion_matrix(pred_labels_first, pred_labels_sec, unclass_detect_first,
                                                     unclass_detect_sec)
            return conf_mat

    def _uncertainty(self, uncertainty_method):
        pass

    def _bc_pred_delta(self):
        mean_bc_first_first = calc_bray_curtis_dissimilarity(self.first_data, self.first_data, self_cohort=True)
        mean_bc_sec_first = calc_bray_curtis_dissimilarity(self.sec_data, self.first_data, self_cohort=False)
        delta_bc_first = np.abs(mean_bc_first_first - mean_bc_sec_first)
        pred_labels_first = np.array([1 if x < y else 0 for x, y in zip(mean_bc_first_first, mean_bc_sec_first)])

        mean_bc_sec_sec = calc_bray_curtis_dissimilarity(self.sec_data, self.sec_data, self_cohort=True)
        mean_bc_first_sec = calc_bray_curtis_dissimilarity(self.first_data, self.sec_data, self_cohort=False)
        delta_bc_sec = np.abs(mean_bc_sec_sec - mean_bc_first_sec)
        pred_labels_sec = np.array([0 if x < y else 1 for x, y in zip(mean_bc_sec_sec, mean_bc_first_sec)])
        return delta_bc_first, delta_bc_sec, pred_labels_first, pred_labels_sec

    def _idoa_pred_delta(self):
        idoa_obj_first_first = IDOA(self.first_data, self.first_data, method='min_max_zero')
        idoa_first_first = idoa_obj_first_first.calc_idoa_vector()
        idoa_obj_sec_first = IDOA(self.sec_data, self.first_data, method='min_max_zero')
        idoa_sec_first = idoa_obj_sec_first.calc_idoa_vector()
        delta_IDOA_first = np.abs(idoa_first_first - idoa_sec_first)
        pred_labels_first = np.array([1 if x < y else 0 for x, y in zip(idoa_first_first, idoa_sec_first)])

        idoa_obj_sec_sec = IDOA(self.sec_data, self.sec_data, method='min_max_zero')
        idoa_sec_sec = idoa_obj_sec_sec.calc_idoa_vector()
        idoa_obj_first_sec = IDOA(self.first_data, self.sec_data, method='min_max_zero')
        idoa_first_sec = idoa_obj_first_sec.calc_idoa_vector()
        delta_IDOA_sec = np.abs(idoa_sec_sec - idoa_first_sec)
        pred_labels_sec = np.array([0 if x < y else 1 for x, y in zip(idoa_sec_sec, idoa_first_sec)])
        return delta_IDOA_first, delta_IDOA_sec, pred_labels_first, pred_labels_sec

    @staticmethod
    def _cals_unclassified(delta_first, delta_sec, uncertainty_method):
        delta_tot = np.concatenate((delta_first, delta_sec))
        if uncertainty_method == 'std':
            std = np.std(delta_tot)
            mask_first = delta_first >= std
            unclass_detect_first = np.where(mask_first, 1, np.nan)
            mask_sec = delta_sec >= std
            unclass_detect_sec = np.where(mask_sec, 0, np.nan)
            return unclass_detect_first, unclass_detect_sec
        elif uncertainty_method == 'percentile':
            per = 0.8
            delta_sorted = np.sort(delta_tot)[::-1]
            cutoff_index = int(per * len(delta_sorted))
            cutoff_value = delta_sorted[cutoff_index]
            mask_first = delta_first >= cutoff_value
            unclass_detect_first = np.where(mask_first, 1, np.nan)
            mask_sec = delta_sec >= cutoff_value
            unclass_detect_sec = np.where(mask_sec, 0, np.nan)
            return unclass_detect_first, unclass_detect_sec

    def _create_confusion_matrix(self, pred_labels_first, pred_labels_sec, unclass_detect_first,
                                                     unclass_detect_sec):
        tp_mask = (pred_labels_first == self.first_data_labels) & (pred_labels_first == unclass_detect_first)
        tp = np.sum(tp_mask)
        unclass_first_mask = np.isnan(unclass_detect_first)
        unclass_first = np.count_nonzero(unclass_first_mask)
        fn = self.first_data.shape[0] - (tp + unclass_first)

        tn_mask = (pred_labels_sec == self.sec_data_labels) & (pred_labels_sec == unclass_detect_sec)
        tn = np.sum(tn_mask)
        unclass_sec_mask = np.isnan(unclass_detect_sec)
        unclass_sec = np.count_nonzero(unclass_sec_mask)
        fp = self.sec_data.shape[0] - (tn + unclass_sec)

        return np.array([[tp, fp], [fn, tn], [unclass_first, unclass_sec]])
"""

####################################################
#p_vals_vector_bc = calc_p_value(ASD_data_norm, control_data_norm, num_iters=100, method='braycurtis')
#p_vals_vector_idoa = calc_p_value(ASD_data_norm, control_data_norm, num_iters=100, method='IDOA')
#tic = time.time()
#pred_labels, p_valus, conf_mat = classify(control_data_norm, ASD_data_norm, 10, 'IDOA', 'KS')
#toc = time.time()
#print(toc-tic)
#classifier_obj_BC_KS = Classifier(control_data_norm, ASD_data_norm, 50, 'braycurtis', 'KS')
#info_BC_KS = classifier_obj_BC_KS.classify()
#p_values_BC_KS = info_BC_KS['p values']

#classifier_obj_BC_MW = Classifier(control_data_norm, ASD_data_norm, 50, 'braycurtis', 'MW')
#info_BC_MW = classifier_obj_BC_MW.classify()
#p_values_BC_MW = info_BC_MW['p values']
#pred_labels_BC_MW = info_BC_MW['pred labels']
#real_lables = np.hstack((np.ones(control_data_norm.shape[0]), np.zeros(ASD_data_norm.shape[0])))
#correct_ind_BC = np.where(pred_labels_BC_MW == real_lables)[0]
#incorrect_ind_BC = np.where(pred_labels_BC_MW != real_lables)[0]
#p_values_BC_MW_scaled = -np.log10(p_values_BC_MW)
#p_values_BC_MW_scaled_correct = p_values_BC_MW_scaled[correct_ind_BC]
#p_values_BC_MW_scaled_incorrect = p_values_BC_MW_scaled[incorrect_ind_BC]

#classifier_obj_IDOA_KS = Classifier(control_data_norm, ASD_data_norm, 50, 'IDOA', 'KS')
#info_IDOA_KS = classifier_obj_IDOA_KS.classify()
#p_values_IDOA_KS = info_IDOA_KS['p values']

#classifier_obj_IDOA_MW = Classifier(control_data_norm, ASD_data_norm, 50, 'IDOA', 'MW')
#info_IDOA_MW = classifier_obj_IDOA_MW.classify()
#p_values_IDOA_MW = info_IDOA_MW['p values']
#pred_labels_IDOA_MW = info_IDOA_MW['pred labels']
#correct_ind_IDOA = np.where(pred_labels_IDOA_MW == real_lables)[0]
#incorrect_ind_IDOA = np.where(pred_labels_IDOA_MW != real_lables)[0]
#p_values_IDOA_MW_scaled = -np.log10(p_values_IDOA_MW)
#p_values_IDOA_MW_scaled_correct = p_values_IDOA_MW_scaled[correct_ind_IDOA]
#p_values_IDOA_MW_scaled_incorrect = p_values_IDOA_MW_scaled[incorrect_ind_IDOA]

#import matplotlib.pyplot as plt

# Create a histogram for the first distribution (data1)
#plt.hist(p_values_IDOA_MW_scaled_correct, bins=30, alpha=0.5, color='blue', label='IDOA correct')

# Create a histogram for the second distribution (data2)
#plt.hist(p_values_BC_MW_scaled_correct, bins=30, alpha=0.5, color='red', label='BC correct')

# Add labels and legend
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.legend()
#plt.show()

# Create a histogram for the first distribution (data1)
#plt.hist(p_values_IDOA_MW_scaled_incorrect, bins=30, alpha=0.5, color='blue', label='IDOA incorrect')

# Create a histogram for the second distribution (data2)
#plt.hist(p_values_BC_MW_scaled_incorrect, bins=30, alpha=0.5, color='red', label='BC incorrect')

# Add labels and legend
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.legend()
#plt.show()


#class_obj = Classifier(control_data_norm, ASD_data_norm)
#conf_mat = class_obj.classify(method='IDOA', uncertainty_method='percentile')
####################################################


"""
# Over cutoff histograms
fig = go.Figure()

fig.add_trace(go.Histogram(x=over_cutoff_asd_control, marker_color='rgba(100, 149, 237, 0.7)'))

# Add a black vertical line at x=143
fig.add_shape(
    go.layout.Shape(
        type='line',
        x0=143,
        x1=143,
        y0=0,
        y1=max(over_cutoff_asd_control),
        line=dict(color='black', width=2)
    )
)

# Customize layout
fig.update_layout(
    width=600,
    height=600,
    xaxis_title='Over cutoff points',
    yaxis_title='Counts',
    title='Control samples w.r.t ASD cohort',
    #title_text='',
    font=dict(
        family='latex',
        size=25,
    ),
    xaxis=dict(
        tickfont=dict(
            size=20,
        ),
    ),
    yaxis=dict(
        tickfont=dict(
            size=20,
        ),
    ),
    xaxis_showgrid=False,
    yaxis_showgrid=False,
)

# Show the plot
fig.show()

fig = go.Figure()

fig.add_trace(go.Histogram(x=over_cutoff_asd_asd, marker_color='rgba(100, 149, 237, 0.7)'))

# Add a black vertical line at x=143
fig.add_shape(
    go.layout.Shape(
        type='line',
        x0=142,
        x1=142,
        y0=0,
        y1=max(over_cutoff_asd_asd),
        line=dict(color='black', width=2)
    )
)

# Customize layout
fig.update_layout(
    width=600,
    height=600,
    xaxis_title='Over cutoff points',
    yaxis_title='Counts',
    title='ASD samples w.r.t ASD cohort',
    #title_text='',
    font=dict(
        family='latex',
        size=25,
    ),
    xaxis=dict(
        tickfont=dict(
            size=20,
        ),
    ),
    yaxis=dict(
        tickfont=dict(
            size=20,
        ),
    ),
    xaxis_showgrid=False,
    yaxis_showgrid=False,
)

# Show the plot
fig.show()

fig = go.Figure()

fig.add_trace(go.Histogram(x=over_cutoff_control_control, marker_color='rgba(100, 149, 237, 0.7)'))

# Add a black vertical line at x=143
fig.add_shape(
    go.layout.Shape(
        type='line',
        x0=110,
        x1=110,
        y0=0,
        y1=max(over_cutoff_control_control),
        line=dict(color='black', width=2)
    )
)

# Customize layout
fig.update_layout(
    width=600,
    height=600,
    xaxis_title='Over cutoff points',
    yaxis_title='Counts',
    title='Control samples w.r.t control cohort',
    #title_text='',
    font=dict(
        family='latex',
        size=25,
    ),
    xaxis=dict(
        tickfont=dict(
            size=20,
        ),
    ),
    yaxis=dict(
        tickfont=dict(
            size=20,
        ),
    ),
    xaxis_showgrid=False,
    yaxis_showgrid=False,
)

# Show the plot
fig.show()

fig = go.Figure()

fig.add_trace(go.Histogram(x=over_cutoff_control_asd, marker_color='rgba(100, 149, 237, 0.7)'))

# Add a black vertical line at x=143
fig.add_shape(
    go.layout.Shape(
        type='line',
        x0=111,
        x1=111,
        y0=0,
        y1=max(over_cutoff_control_asd),
        line=dict(color='black', width=2)
    )
)

# Customize layout
fig.update_layout(
    width=600,
    height=600,
    xaxis_title='Over cutoff points',
    yaxis_title='Counts',
    title='ASD samples w.r.t control cohort',
    #title_text='',
    font=dict(
        family='latex',
        size=25,
    ),
    xaxis=dict(
        tickfont=dict(
            size=20,
        ),
    ),
    yaxis=dict(
        tickfont=dict(
            size=20,
        ),
    ),
    xaxis_showgrid=False,
    yaxis_showgrid=False,
)

# Show the plot
fig.show()
"""

#confusion_matrix = calculate_confusion_matrix(dist_control_control_vector_asd,
#                                              dist_asd_control_vector,
#                                              dist_asd_asd_vector,
#                                              dist_control_asd_vector)
#print(confusion_matrix)
#g = 1
# Calculate the accuracy (success rate)
#accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix)
#print("Accuracy (Success Rate): {:.2%}".format(accuracy))

#confusion_matrix_dist = calculate_confusion_matrix(dist_control_control_vector_asd,
#                                                   dist_asd_control_vector,
#                                                   dist_asd_asd_vector,
#                                                   dist_control_asd_vector)
#print(confusion_matrix_dist)

# Calculate the accuracy (success rate)
#accuracy_dist = (confusion_matrix_dist[0, 0] + confusion_matrix_dist[1, 1]) /\
#                np.sum(confusion_matrix_dist)
#print("Accuracy (Success Rate): {:.2%}".format(accuracy_dist))

#import plotly.express as px

# Create a scatter plot using Plotly
#fig = px.scatter(x=sm_x_asd, y=sm_y_asd, title="LOWESS Smoother")

# Show the Plotly figure
#fig.show()

# Create a scatter plot using Plotly
#fig = px.scatter(x=sm_x_control, y=sm_y_control, title="LOWESS Smoother")

# Show the Plotly figure
#fig.show()

#def accuracy_vs_cutoff(cutoff_range, delta=0.03):
#    """
#    Calculate the accuracy of the IDOA method for different cutoff values
#    :param cutoff_range: list type, range of cutoff values of the form [min, max]
#    :param delta: float type, step size for the cutoff values
#    :return: accuracy_vals: list type, accuracy values for the different cutoff values
#             cutoff_vals: list type, cutoff values
#    """
#    # Initialize the lists
#    cutoff_vals = np.arange(cutoff_range[0], cutoff_range[1], delta)
#    accuracy_vals = []
#    # Calculate the accuracy for each cutoff value
#    for cutoff in cutoff_vals:
#        idoa_control_asd_vector_object = IDOA(control_data_norm, ASD_data_norm, identical=False,
#                                              percentage=per, min_num_points=min_num_points,
#                                              min_overlap=cutoff, max_overlap=maximal,
#                                              zero_overlap=zero, method='min_max_zero')
#        idoa_control_asd_vector = idoa_control_asd_vector_object.calc_idoa_vector()
#        idoa_asd_control_vector_object = IDOA(ASD_data_norm, control_data_norm, identical=False,
#                                              percentage=per, min_num_points=min_num_points,
#                                              min_overlap=cutoff, max_overlap=maximal,
#                                              zero_overlap=zero, method='min_max_zero')
#        idoa_asd_control_vector = idoa_asd_control_vector_object.calc_idoa_vector()
#        idoa_asd_asd_vector_object = IDOA(ASD_data_norm, ASD_data_norm, identical=True,
#                                          percentage=per, min_num_points=min_num_points,
#                                          min_overlap=cutoff, max_overlap=maximal,
#                                          zero_overlap=zero, method='min_max_zero')
#        idoa_asd_asd_vector = idoa_asd_asd_vector_object.calc_idoa_vector()
#        idoa_control_control_vector_asd_object = IDOA(control_data_norm, control_data_norm,
#                                                      identical=True, percentage=per,
#                                                      min_num_points=min_num_points,
#                                                      min_overlap=cutoff, max_overlap=maximal,
#                                                      zero_overlap=zero, method='min_max_zero')
#        idoa_control_control_vector_asd = idoa_control_control_vector_asd_object.calc_idoa_vector()
#
#        confusion_matrix = calculate_confusion_matrix(idoa_control_control_vector_asd,
#                                                      idoa_asd_control_vector,
#                                                      idoa_asd_asd_vector,
#                                                      idoa_control_asd_vector)
#
#        accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix)
#
#        accuracy_vals.append(accuracy)
#
#    return cutoff_vals, accuracy_vals

#def accuracy_vs_cutoff_percentage(cutoff_range, delta=0.1):
#    """
#    Calculate the accuracy of the IDOA method for different cutoff values (percentage)
#    :param cutoff_range: list type, range of cutoff values of the form [min, max]
#    :param delta: float type, step size for the cutoff values
#    :return: accuracy_vals: list type, accuracy values for the different cutoff values
#             cutoff_vals: list type, cutoff values
#    """
#    # Initialize the lists
#    cutoff_vals = np.arange(cutoff_range[0], cutoff_range[1], delta)
#    accuracy_vals = []
#    # Calculate the accuracy for each cutoff value
#    for cutoff in cutoff_vals:
#        cutoff = float(cutoff)
#        print(cutoff)
#        idoa_control_asd_vector_object = IDOA(control_data_norm, ASD_data_norm, identical=False,
#                                              percentage=cutoff, min_num_points=min_num_points,
#                                              min_overlap=min_overlap_control, max_overlap=maximal,
#                                              zero_overlap=zero, method='percentage')
#        idoa_control_asd_vector = idoa_control_asd_vector_object.calc_idoa_vector()
#        idoa_asd_control_vector_object = IDOA(ASD_data_norm, control_data_norm, identical=False,
#                                              percentage=cutoff, min_num_points=min_num_points,
#                                              min_overlap=min_overlap_ASD, max_overlap=maximal,
#                                              zero_overlap=zero, method='percentage')
#        idoa_asd_control_vector = idoa_asd_control_vector_object.calc_idoa_vector()
#        idoa_asd_asd_vector_object = IDOA(ASD_data_norm, ASD_data_norm, identical=True,
#                                          percentage=cutoff, min_num_points=min_num_points,
#                                          min_overlap=min_overlap_ASD, max_overlap=maximal,
#                                          zero_overlap=0, method='percentage')
#        idoa_asd_asd_vector = idoa_asd_asd_vector_object.calc_idoa_vector()
#        idoa_control_control_vector_asd_object = IDOA(control_data_norm, control_data_norm,
#                                                      identical=True, percentage=cutoff,
#                                                      min_num_points=min_num_points,
#                                                      min_overlap=min_overlap_control, max_overlap=maximal,
#                                                      zero_overlap=0, method='percentage')
#        idoa_control_control_vector_asd = idoa_control_control_vector_asd_object.calc_idoa_vector()
#
#        confusion_matrix = calculate_confusion_matrix(idoa_control_control_vector_asd,
#                                                      idoa_asd_control_vector,
#                                                      idoa_asd_asd_vector,
#                                                      idoa_control_asd_vector)
#
#        accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix)
#
#        accuracy_vals.append(accuracy)
#
#    return cutoff_vals, accuracy_vals

# Create the Accuracy vs. Cutoff plots for percentage method and min_max_zero method
#cutoff_vals, accuracy_vals = accuracy_vs_cutoff([min_cut, max_cut],  step_cut)

#cutoff_vals_per, accuracy_vals_per = accuracy_vs_cutoff_percentage([min_per, max_per],  step_per)

"""
# Add trace for accuracy values
fig = go.Figure()
fig_per = go.Figure()

#fig_per.add_trace(go.Scatter(x=100 - cutoff_vals_per, y=accuracy_vals_per, mode='markers',
#                             name='IDOA', marker=dict(color='red')))
#fig.add_trace(go.Scatter(x=cutoff_vals, y=accuracy_vals, mode='markers', name='IDOA',
#                         marker=dict(color='red')))

#fig_per.add_trace(go.Scatter(x=[min(100 - cutoff_vals), max(100 - cutoff_vals)],
#                             y=[accuracy_dist, accuracy_dist], mode='lines',
#                             line=dict(color="blue", width=2), name='Distances'))

#fig.add_trace(go.Scatter(x=[min(cutoff_vals), max(cutoff_vals)], y=[accuracy_dist, accuracy_dist],
#                         mode='lines', line=dict(color="blue", width=2), name='Distances'))

# Update layout
fig.update_layout(
    width=600,
    height=600,
    xaxis_title="Cutoff",
    yaxis_title="Accuracy",
    font=dict(family='Computer Modern'),
    legend=dict(
        x=0.02,
        y=0.98,
        font=dict(family='Computer Modern', size=20),
    ),
    showlegend=True,
    xaxis=dict(
        showgrid=False,
        linecolor='black',
        tickfont=dict(size=20),
        titlefont=dict(size=25),
    ),
    yaxis=dict(
        showgrid=False,
        linecolor='black',
        tickfont=dict(size=20),
        titlefont=dict(size=20),
    ),
    plot_bgcolor='white',
)

# Show the plot
#fig.show()

# Update layout
fig_per.update_layout(
    width=600,
    height=600,
    xaxis_title="Percentage",
    yaxis_title="Accuracy",
    font=dict(family='Computer Modern'),
    legend=dict(
        x=0.02,
        y=0.98,
        font=dict(family='Computer Modern', size=20),
    ),
    showlegend=True,
    xaxis=dict(
        showgrid=False,
        linecolor='black',
        tickfont=dict(size=20),
        titlefont=dict(size=25),
    ),
    yaxis=dict(
        showgrid=False,
        linecolor='black',
        tickfont=dict(size=20),
        titlefont=dict(size=20),
    ),
    plot_bgcolor='white',
)

# Show the plot
#fig_per.show()
"""

# Create the ROC curves for IDOA and distance
#binary_vector = np.concatenate((np.ones(111), np.zeros(143)))
#delta_vector = np.concatenate((idoa_asd_control_vector - idoa_control_control_vector_asd,
#                               idoa_asd_asd_vector - idoa_control_asd_vector))
#delta_vector_dist = np.concatenate((dist_asd_control_vector - dist_control_control_vector_asd,
#                                    dist_asd_asd_vector - dist_control_asd_vector))

#def calculate_roc_curve(binary_vector, delta_vector_method_first, delta_vector_method_sec,
#                        num_rows, epsilon):
#    """
#    Calculates the ROC curve for IDOA and distances
#    :param binary_vector: Binary vector with 1 for ASD and 0 for control
#    :param delta_vector_method_first: Vector with the differences between the IDOA values
#    :param delta_vector_method_sec: Vector with the differences between the distance values
#    :param num_rows: Number of rows for the threshold matrix
#    :param epsilon: Epsilon value for the threshold matrix
#    """
    # Construct the threshold matrices
#    thresholds_first = construct_threshold_matrix(delta_vector_method_first, num_rows, epsilon)
#    thresholds_sec = construct_threshold_matrix(delta_vector_method_sec, num_rows, epsilon)

    # Calculate True Positive (TP) and False Positive (FP) rates for each threshold
#    tpr_list_first = []
#    fpr_list_first = []
#    tpr_list_sec = []
#    fpr_list_sec = []

#    for threshold_first, threshold_sec in zip(thresholds_first, thresholds_sec):
#        # Predicted labels based on the threshold
#        predicted_labels_first = np.where(threshold_first > 0, 1, 0)
#        predicted_labels_sec = np.where(threshold_sec > 0, 1, 0)

        # True Positives (TP) and False Positives (FP)
#        tp_first = np.sum((predicted_labels_first == 1) & (binary_vector == 1))
#        fp_first = np.sum((predicted_labels_first == 1) & (binary_vector == 0))
#        tp_sec = np.sum((predicted_labels_sec == 1) & (binary_vector == 1))
#        fp_sec = np.sum((predicted_labels_sec == 1) & (binary_vector == 0))

        # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
#        tpr_first = tp_first / np.sum(binary_vector == 1)
#        fpr_first = fp_first / np.sum(binary_vector == 0)
#        tpr_sec = tp_sec / np.sum(binary_vector == 1)
#        fpr_sec = fp_sec / np.sum(binary_vector == 0)

#        tpr_list_first.append(tpr_first)
#        fpr_list_first.append(fpr_first)
#        tpr_list_sec.append(tpr_sec)
#        fpr_list_sec.append(fpr_sec)

#    auc_first = np.trapz(tpr_list_first, fpr_list_first)
#    auc_sec = np.trapz(tpr_list_sec, fpr_list_sec)

#    trace1 = go.Scatter(x=fpr_list_first, y=tpr_list_first, mode='lines', name='IDOA', line=dict(color='red', width=2))
#    trace2 = go.Scatter(x=fpr_list_sec, y=tpr_list_sec, mode='lines', name='Distances', line=dict(color='blue', width=2))

#    layout = go.Layout(
#        width=600,
#        height=600,
#        xaxis=dict(
#            title='FPR',
#            titlefont=dict(family='Computer Modern', size=25),
#            tickfont=dict(family='Computer Modern', size=20),
#            showgrid=False,
#            linecolor='black',
#        ),
#        yaxis=dict(
#            title='TPR',
#            titlefont=dict(family='Computer Modern', size=25),  #
#            tickfont=dict(family='Computer Modern', size=20),
#            showgrid=False,
#            linecolor='black',
#        ),
#        legend=dict(
#            font=dict(family='Computer Modern', size=20),
#            x=0.02,
#            y=0.98
#        ),
#        showlegend=True,
#        margin=dict(l=50, r=50, b=50, t=50),
#        plot_bgcolor='white',
#    )

#    fig = go.Figure(data=[trace1, trace2], layout=layout)

#    fig.add_annotation(
#        text=f'AUC: {auc_first:.3f}',
#        x=0.95,
#        y=0.1,
#        xref='paper',
#        yref='paper',
#        showarrow=False,
#        font=dict(family='Computer Modern', size=12, color='red'),
#        align='right',
#    )

#        text=f'AUC: {auc_sec:.3f}',
#        x=0.95,
#        y=0.05,
#        xref='paper',
#        yref='paper',
#        showarrow=False,
#        font=dict(family='Computer Modern', size=12, color='blue'),
#        align='right',
#    )
#    fig.show()

# Create ROC curve plot
#calculate_roc_curve(binary_vector, delta_vector, delta_vector_dist, 2000, 0.0001)

# Load already calculated IDOA values (cutoff = 0.5)
#df_idoa_control_asd_vector_05 = pd.read_csv('idoa_control_ASD_vector.csv', header=None)
#idoa_control_asd_vector_05 = df_idoa_control_asd_vector_05.to_numpy()
#idoa_control_asd_vector_05 = idoa_control_asd_vector_05.flatten()
#df_idoa_control_control_vector_asd_05 = pd.read_csv('idoa_control_control_vector.csv', header=None)
#idoa_control_control_vector_asd_05 = df_idoa_control_control_vector_asd_05.to_numpy()
#idoa_control_control_vector_asd_05 = idoa_control_control_vector_asd_05.flatten()
#df_idoa_asd_control_vector_05 = pd.read_csv('idoa_ASD_control_vector.csv', header=None)
#idoa_asd_control_vector_05 = df_idoa_asd_control_vector_05.to_numpy()
#idoa_asd_control_vector_05 = idoa_asd_control_vector_05.flatten()
#df_idoa_asd_asd_vector_05 = pd.read_csv('idoa_asd_asd_vector.csv', header=None)
#idoa_asd_asd_vector_05 = df_idoa_asd_asd_vector_05.to_numpy()
#idoa_asd_asd_vector_05 = idoa_asd_asd_vector_05.flatten()

#df_CM_asd_idoa = pd.read_csv('con_mat_IDOA.csv', header=None)
#CM_asd_idoa = df_CM_asd_idoa.to_numpy()
#df_CM_asd_dist = pd.read_csv('con_mat_distances.csv', header=None)
#CM_asd_dist = df_CM_asd_dist.to_numpy()

#over_cutoff_control_control = idoa_control_control_vector_asd_object.over_cutoff
#over_cutoff_asd_control = idoa_asd_control_vector_object.over_cutoff