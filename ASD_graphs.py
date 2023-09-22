import os
from dash import Dash, html, dcc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import MDS
from scipy.spatial.distance import cdist
from Functions import idoa, normalize_data
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from IDOA import IDOA

########## Load data - ASD ##########
os.chdir(r'C:\Users\shaya\OneDrive\Desktop\IDOA\ASD_results\ASD_measures')
df_dist_control_asd_vector = pd.read_csv('dist_control_ASD_vector.csv', header=None)
dist_control_asd_vector = df_dist_control_asd_vector.to_numpy()
dist_control_asd_vector = dist_control_asd_vector.flatten()
df_dist_control_control_vector_asd = pd.read_csv('dist_control_control_vector.csv', header=None)
dist_control_control_vector_asd = df_dist_control_control_vector_asd.to_numpy()
dist_control_control_vector_asd = dist_control_control_vector_asd.flatten()
df_dist_asd_control_vector = pd.read_csv('dist_ASD_control_vector.csv', header=None)
dist_asd_control_vector = df_dist_asd_control_vector.to_numpy()
dist_asd_control_vector = dist_asd_control_vector.flatten()
df_dist_asd_asd_vector = pd.read_csv('dist_ASD_ASD_vector.csv', header=None)
dist_asd_asd_vector = df_dist_asd_asd_vector.to_numpy()
dist_asd_asd_vector = dist_asd_asd_vector.flatten()

#df_idoa_control_asd_vector = pd.read_csv('idoa_control_ASD_vector.csv', header=None)
#idoa_control_asd_vector = df_idoa_control_asd_vector.to_numpy()
#idoa_control_asd_vector = idoa_control_asd_vector.flatten()
#df_idoa_control_control_vector_asd = pd.read_csv('idoa_control_control_vector.csv', header=None)
#idoa_control_control_vector_asd = df_idoa_control_control_vector_asd.to_numpy()
#idoa_control_control_vector_asd = idoa_control_control_vector_asd.flatten()
#df_idoa_asd_control_vector = pd.read_csv('idoa_ASD_control_vector.csv', header=None)
#idoa_asd_control_vector = df_idoa_asd_control_vector.to_numpy()
#idoa_asd_control_vector = idoa_asd_control_vector.flatten()
#df_idoa_asd_asd_vector = pd.read_csv('idoa_asd_asd_vector.csv', header=None)
#idoa_asd_asd_vector = df_idoa_asd_asd_vector.to_numpy()
#idoa_asd_asd_vector = idoa_asd_asd_vector.flatten()

df_DOC_control_asd = pd.read_csv('Doc_mat_control.csv', header=None)
DOC_control_asd = df_DOC_control_asd.to_numpy()
df_DOC_asd = pd.read_csv('Doc_mat_ASD.csv', header=None)
DOC_asd = df_DOC_asd.to_numpy()
df_CM_asd_idoa = pd.read_csv('con_mat_IDOA.csv', header=None)
CM_asd_idoa = df_CM_asd_idoa.to_numpy()
df_CM_asd_dist = pd.read_csv('con_mat_distances.csv', header=None)
CM_asd_dist = df_CM_asd_dist.to_numpy()

# IDOA of some ASD sample
ASD_data = pd.read_csv('ASD_data.csv', header=None)
ASD_data = ASD_data.to_numpy()
control_data = pd.read_csv('control_data.csv', header=None)
control_data = control_data.to_numpy()

def normalize_cohort(cohort):
    # normalization function
    if cohort.ndim == 1:
        cohort_normalized = cohort / cohort.sum()
    else:
        cohort_normalized = cohort / np.linalg.norm(cohort, ord=1, axis=1, keepdims=True)
    return cohort_normalized

ASD_data_norm = normalize_cohort(ASD_data.T)
control_data_norm = normalize_cohort(control_data.T)

idoa_control_asd_vector_object = IDOA(control_data_norm, ASD_data_norm, identical=False, percentage=5,
                                      min_num_points=0, min_overlap=0.954,
                                      max_overlap=1, zero_overlap=0, method='percentage')
idoa_control_asd_vector = idoa_control_asd_vector_object.calc_idoa_vector()
idoa_asd_control_vector_object = IDOA(ASD_data_norm, control_data_norm, identical=False, percentage=5,
                                      min_num_points=0, min_overlap=0.954,
                                      max_overlap=1, zero_overlap=0, method='percentage')
idoa_asd_control_vector = idoa_asd_control_vector_object.calc_idoa_vector()
idoa_asd_asd_vector_object = IDOA(ASD_data_norm, ASD_data_norm, identical=True, percentage=5,
                                  min_num_points=0, min_overlap=0.954,
                                  max_overlap=1, zero_overlap=0, method='percentage')
idoa_asd_asd_vector = idoa_asd_asd_vector_object.calc_idoa_vector()
idoa_control_control_vector_asd_object = IDOA(control_data_norm, control_data_norm, identical=True, percentage=5,
                                              min_num_points=0, min_overlap=0.954,
                                              max_overlap=1, zero_overlap=0, method='percentage')
idoa_control_control_vector_asd = idoa_control_control_vector_asd_object.calc_idoa_vector()

def calculate_confusion_matrix(control_control, asd_control, asd_asd,
                               control_asd):
    m = len(control_control)
    n = len(asd_asd)

    healthy_count = np.sum(control_control < asd_control)
    asd_count = np.sum(asd_asd < control_asd)

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

confusion_matrix = calculate_confusion_matrix(idoa_control_control_vector_asd, idoa_asd_control_vector,
                                              idoa_asd_asd_vector, idoa_control_asd_vector)
print(confusion_matrix)

# Calculate the accuracy (success rate)
accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix)
print("Accuracy (Success Rate): {:.2%}".format(accuracy))

confusion_matrix_dist = calculate_confusion_matrix(dist_control_control_vector_asd, dist_asd_control_vector,
                                              dist_asd_asd_vector, dist_control_asd_vector)
print(confusion_matrix_dist)

# Calculate the accuracy (success rate)
accuracy_dist = (confusion_matrix_dist[0, 0] + confusion_matrix_dist[1, 1]) / np.sum(confusion_matrix_dist)
print("Accuracy (Success Rate): {:.2%}".format(accuracy_dist))

def accuracy_vs_cutoff(cutoff_range, delta=0.03):
    cutoff_vals = np.arange(cutoff_range[0], cutoff_range[1], delta)
    accuracy_vals = []
    for cutoff in cutoff_vals:
        idoa_control_asd_vector_object = IDOA(control_data_norm, ASD_data_norm, identical=False, percentage=60,
                                              min_num_points=0, min_overlap=cutoff,
                                              max_overlap=1, zero_overlap=0, method='min_max_zero')
        idoa_control_asd_vector = idoa_control_asd_vector_object.calc_idoa_vector()
        idoa_asd_control_vector_object = IDOA(ASD_data_norm, control_data_norm, identical=False, percentage=60,
                                              min_num_points=0, min_overlap=cutoff,
                                              max_overlap=1, zero_overlap=0, method='min_max_zero')
        idoa_asd_control_vector = idoa_asd_control_vector_object.calc_idoa_vector()
        idoa_asd_asd_vector_object = IDOA(ASD_data_norm, ASD_data_norm, identical=True, percentage=60, min_num_points=0,
                                          min_overlap=cutoff,
                                          max_overlap=1, zero_overlap=0, method='min_max_zero')
        idoa_asd_asd_vector = idoa_asd_asd_vector_object.calc_idoa_vector()
        idoa_control_control_vector_asd_object = IDOA(control_data_norm, control_data_norm, identical=True, percentage=60,
                                                      min_num_points=0, min_overlap=cutoff,
                                                      max_overlap=1, zero_overlap=0, method='min_max_zero')
        idoa_control_control_vector_asd = idoa_control_control_vector_asd_object.calc_idoa_vector()

        confusion_matrix = calculate_confusion_matrix(idoa_control_control_vector_asd, idoa_asd_control_vector,
                                                      idoa_asd_asd_vector, idoa_control_asd_vector)

        accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix)

        accuracy_vals.append(accuracy)

    return cutoff_vals, accuracy_vals

def accuracy_vs_cutoff_percentage(cutoff_range, delta=0.1):
    cutoff_vals = np.arange(cutoff_range[0], cutoff_range[1], delta)
    accuracy_vals = []
    for cutoff in cutoff_vals:
        print(cutoff)
        idoa_control_asd_vector_object = IDOA(control_data_norm, ASD_data_norm, identical=False, percentage=cutoff,
                                              min_num_points=0, min_overlap=0.5,
                                              max_overlap=1, zero_overlap=0, method='percentage')
        idoa_control_asd_vector = idoa_control_asd_vector_object.calc_idoa_vector()
        idoa_asd_control_vector_object = IDOA(ASD_data_norm, control_data_norm, identical=False, percentage=cutoff,
                                              min_num_points=0, min_overlap=0.5,
                                              max_overlap=1, zero_overlap=0, method='percentage')
        idoa_asd_control_vector = idoa_asd_control_vector_object.calc_idoa_vector()
        idoa_asd_asd_vector_object = IDOA(ASD_data_norm, ASD_data_norm, identical=True, percentage=cutoff, min_num_points=0,
                                          min_overlap=0.5,
                                          max_overlap=1, zero_overlap=0, method='percentage')
        idoa_asd_asd_vector = idoa_asd_asd_vector_object.calc_idoa_vector()
        idoa_control_control_vector_asd_object = IDOA(control_data_norm, control_data_norm, identical=True, percentage=cutoff,
                                                      min_num_points=0, min_overlap=0.5,
                                                      max_overlap=1, zero_overlap=0, method='percentage')
        idoa_control_control_vector_asd = idoa_control_control_vector_asd_object.calc_idoa_vector()

        confusion_matrix = calculate_confusion_matrix(idoa_control_control_vector_asd, idoa_asd_control_vector,
                                                      idoa_asd_asd_vector, idoa_control_asd_vector)

        accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix)

        accuracy_vals.append(accuracy)

    return cutoff_vals, accuracy_vals

cutoff_vals, accuracy_vals = accuracy_vs_cutoff_percentage([1, 20],  0.5)

# Add trace for accuracy values
fig = go.Figure()

# Add trace for accuracy values
fig.add_trace(go.Scatter(x=100 - cutoff_vals, y=accuracy_vals, mode='markers', name='IDOA', marker=dict(color='red')))

# Add a vertical line in blue
#fig.add_shape(
#    type="line",
#    x0=min(cutoff_vals), x1=max(cutoff_vals),
#    y0=accuracy_dist, y1=accuracy_dist,
#    line=dict(color="blue", width=2),
#    name='Distances',
#)

fig.add_trace(go.Scatter(x=[min(100 - cutoff_vals), max(100 - cutoff_vals)], y=[accuracy_dist, accuracy_dist],
                         mode='lines', line=dict(color="blue", width=2), name='Distances'))

# Update layout
fig.update_layout(
    width=600,
    height=600,
    xaxis_title="Percentage",
    yaxis_title="Accuracy",
    font=dict(family='Computer Modern'),
    legend=dict(
        x=0.02,
        y=0.98,
        font=dict(family='Computer Modern'),
    ),
    showlegend=True,
    xaxis=dict(
        showgrid=False,  # Remove grid
        linecolor='black',  # Set x-axis line color to black
    ),
    yaxis=dict(
        showgrid=False,  # Remove grid
        linecolor='black',  # Set y-axis line color to black
    ),
    plot_bgcolor='white',  # Set background color to white
)

# Show the plot
fig.show()

binary_vector = np.concatenate((np.ones(111), np.zeros(143)))
delta_vector = np.concatenate((idoa_asd_control_vector - idoa_control_control_vector_asd,
                               idoa_asd_asd_vector - idoa_control_asd_vector))

delta_vector_dist = np.concatenate((dist_asd_control_vector - dist_control_control_vector_asd,
                                    dist_asd_asd_vector - dist_control_asd_vector))

def calculate_roc_curve(binary_vector, delta_vector_method_first, delta_vector_method_sec, num_rows, epsilon):

    def construct_threshold_matrix(delta_vector, num_rows, epsilon):
        max_delta = max(delta_vector)
        min_delta = min(delta_vector)

        # Create a matrix with num_rows rows and columns based on delta_vector
        threshold_matrix = np.tile(delta_vector, (num_rows, 1))

        # Create a linspace for the additional values
        additional_values = np.linspace(-max_delta - epsilon, -min_delta + epsilon, num_rows).reshape(-1, 1)

        # Add the additional values to each row of the matrix
        threshold_matrix = threshold_matrix + additional_values

        return threshold_matrix

    thresholds_first = construct_threshold_matrix(delta_vector_method_first, num_rows, epsilon)
    thresholds_sec = construct_threshold_matrix(delta_vector_method_sec, num_rows, epsilon)

    # Calculate True Positive (TP) and False Positive (FP) rates for each threshold
    tpr_list_first = []
    fpr_list_first = []
    tpr_list_sec = []
    fpr_list_sec = []

    for threshold_first, threshold_sec in zip(thresholds_first, thresholds_sec):
        # Predicted labels based on the threshold
        predicted_labels_first = np.where(threshold_first > 0, 1, 0)
        predicted_labels_sec = np.where(threshold_sec > 0, 1, 0)

        # True Positives (TP) and False Positives (FP)
        tp_first = np.sum((predicted_labels_first == 1) & (binary_vector == 1))
        fp_first = np.sum((predicted_labels_first == 1) & (binary_vector == 0))
        tp_sec = np.sum((predicted_labels_sec == 1) & (binary_vector == 1))
        fp_sec = np.sum((predicted_labels_sec == 1) & (binary_vector == 0))

        # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
        tpr_first = tp_first / np.sum(binary_vector == 1)
        fpr_first = fp_first / np.sum(binary_vector == 0)
        tpr_sec = tp_sec / np.sum(binary_vector == 1)
        fpr_sec = fp_sec / np.sum(binary_vector == 0)

        tpr_list_first.append(tpr_first)
        fpr_list_first.append(fpr_first)
        tpr_list_sec.append(tpr_sec)
        fpr_list_sec.append(fpr_sec)

    # Calculate AUC for each method
    auc_first = np.trapz(tpr_list_first, fpr_list_first)
    auc_sec = np.trapz(tpr_list_sec, fpr_list_sec)

    # Create ROC traces
    trace1 = go.Scatter(x=fpr_list_first, y=tpr_list_first, mode='lines', name='IDOA', line=dict(color='red', width=2))
    trace2 = go.Scatter(x=fpr_list_sec, y=tpr_list_sec, mode='lines', name='Distances', line=dict(color='blue', width=2))

    # Create layout
    layout = go.Layout(
        width=600,
        height=600,
        xaxis=dict(
            title='FPR',
            titlefont=dict(family='Computer Modern'),  # Set title font
            tickfont=dict(family='Computer Modern'),
            showgrid=False,
            linecolor='black',
        ),
        yaxis=dict(
            title='TPR',
            titlefont=dict(family='Computer Modern'),  # Set title font
            tickfont=dict(family='Computer Modern'),
            showgrid=False,
            linecolor='black',
        ),
        legend=dict(
            font=dict(family='Computer Modern'),
            x=0.02,
            y=0.98,
        ),
        showlegend=True,
        margin=dict(l=50, r=50, b=50, t=50),
        plot_bgcolor='white',  # Set background color to white
    )

    # Create figure
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # Add AUC values in the lower right corner
    fig.add_annotation(
        text=f'AUC: {auc_first:.3f}',
        x=0.95,
        y=0.1,
        xref='paper',
        yref='paper',
        showarrow=False,
        font=dict(family='Computer Modern', size=12, color='red'),
        align='right',
    )

    fig.add_annotation(
        text=f'AUC: {auc_sec:.3f}',
        x=0.95,
        y=0.05,
        xref='paper',
        yref='paper',
        showarrow=False,
        font=dict(family='Computer Modern', size=12, color='blue'),
        align='right',
    )

    # Show the plot
    fig.show()

calculate_roc_curve(binary_vector, delta_vector, delta_vector_dist, 2000, 0.0001)

d_o_container_control_asd = idoa_control_asd_vector_object.dissimilarity_overlap_container_no_constraint
d_o_container_asd_control = idoa_asd_control_vector_object.dissimilarity_overlap_container_no_constraint
d_o_container_asd_asd = idoa_asd_asd_vector_object.dissimilarity_overlap_container_no_constraint
d_o_container_control_control = idoa_control_control_vector_asd_object.dissimilarity_overlap_container_no_constraint
"""
import matplotlib.pyplot as plt
for val1, val2, val3, val4 in zip(d_o_container_control_asd[5:15], d_o_container_asd_control[5:15], d_o_container_asd_asd[5:15], d_o_container_control_control[5:15]):

    x1 = val4[0, :]
    y1 = val4[1, :]
    # Create a scatter plot
    plt.scatter(x1, y1, label='Data Points', color='blue', marker='o')

    # Add labels and title
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')
    plt.title('Simple Scatter Plot')

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()
"""
# Initiate overlap and dissimilarity
overlap_vector = np.zeros(np.size(control_data, axis=1))
dissimilarity_vector = np.zeros(np.size(control_data, axis=1))

# Apply IDOA
[new_overlap_vector, new_dissimilarity_vector] = idoa(ASD_data[:, 100], control_data,
                                                      overlap_vector, dissimilarity_vector)
[a, b] = np.polyfit(new_overlap_vector, new_dissimilarity_vector, 1)

app = Dash(__name__)

scatter_idoa_samp_ASD = go.Scattergl(x=new_overlap_vector, y=new_dissimilarity_vector, marker={
    "color": "blue", "size": 5}, showlegend=False, mode="markers")

scatter_DOC_control_asd = go.Scatter(x=DOC_control_asd[1][:], y=DOC_control_asd[0][:], marker={
    "color": "red", "size": 2}, showlegend=False, mode="markers")

sm_x_control_asd, sm_y_control_asd = sm_lowess(endog=DOC_control_asd[0][:], exog=DOC_control_asd[1][:],
                                               frac=0.3, it=3, return_sorted=True).T

sm_x_asd_idoa, sm_y_asd_idoa = sm_lowess(endog=new_dissimilarity_vector, exog=new_overlap_vector,
                                               frac=0.3, it=3, return_sorted=True).T

fig_control_asd = [scatter_DOC_control_asd, go.Scattergl(x=sm_x_control_asd, y=sm_y_control_asd, line={
    "color": "red", "dash": 'solid', "width": 4.5}, showlegend=False), go.Scattergl(
    x=sm_x_asd_idoa, y=sm_y_asd_idoa, line={"color": "blue", "dash": 'solid', "width": 4.5}, showlegend=False)
    , scatter_idoa_samp_ASD]

df_asd_data = pd.read_csv('ASD_data.csv', header=None)
asd_data = df_asd_data.to_numpy()
df_control_data_asd = pd.read_csv('control_data.csv', header=None)
control_data_asd = df_control_data_asd.to_numpy()
asd_data = normalize_data(asd_data)
control_data_asd = normalize_data(control_data_asd)
combined_data = np.concatenate((asd_data.T, control_data_asd.T), axis=0)
dist_mat = cdist(combined_data, combined_data, 'braycurtis')
mds = MDS(n_components=2, metric=True, max_iter=300, random_state=0, dissimilarity='precomputed')
scaled = mds.fit_transform(dist_mat)
num_samples_first = np.size(asd_data, axis=1)

PCoA_asd = go.Scatter(x=scaled[:num_samples_first, 0], y=scaled[:num_samples_first, 1], marker={
    "color": "blue"}, name='ASD', mode="markers")

PCoA_control_asd = go.Scatter(x=scaled[num_samples_first:, 0], y=scaled[num_samples_first:, 1], marker={
    "color": "red"}, name='control', mode="markers")

hist_dist_control_asd_vector, bins_dist_control_asd_vector = np.histogram(dist_control_asd_vector, bins=8,
                                                                          density=True)
hist_dist_control_control_vector, bins_dist_control_control_vector = np.histogram(dist_control_control_vector_asd,
                                                                                  bins=8, density=True)
hist_idoa_control_asd_vector, bins_idoa_control_asd_vector = np.histogram(idoa_control_asd_vector, bins=8,
                                                                          density=True)
hist_idoa_control_control_vector, bins_idoa_control_control_vector = np.histogram(idoa_control_control_vector_asd,
                                                                                  bins=8, density=True)
hist_dist_control_asd = go.Scattergl(x=bins_dist_control_asd_vector, y=hist_dist_control_asd_vector, name='ASD', line={
    "color": "blue"}, fill='tozeroy', fillcolor='blue', opacity=0.5, mode='lines')

hist_dist_control_control = go.Scattergl(x=bins_dist_control_control_vector, y=hist_dist_control_control_vector,
                                         name='Control', line={"color": "red"}, fill='tozeroy', fillcolor='red',
                                         opacity=0.6, mode='lines')

hist_idoa_control_asd = go.Scattergl(x=bins_idoa_control_asd_vector, y=hist_idoa_control_asd_vector, name='ASD',
                                     textfont=dict(size=10), line={"color": "blue"}, fill='tozeroy', fillcolor='blue',
                                     opacity=0.5, mode='lines')

hist_idoa_control_control = go.Scattergl(x=bins_idoa_control_control_vector, y=hist_idoa_control_control_vector,
                                         name='Control', textfont=dict(size=10), line={"color": "red"}, opacity=0.6,
                                         fill='tozeroy', fillcolor='red', mode='lines')

hist_dist_asd_control_vector, bins_dist_asd_control_vector = np.histogram(dist_asd_control_vector, bins=10,
                                                                          density=True)
hist_dist_asd_asd_vector, bins_dist_asd_asd_vector = np.histogram(dist_asd_asd_vector,
                                                                  bins=8, density=True)
hist_idoa_asd_control_vector, bins_idoa_asd_control_vector = np.histogram(idoa_asd_control_vector, bins=10,
                                                                          density=True)
hist_idoa_asd_asd_vector, bins_idoa_asd_asd_vector = np.histogram(idoa_asd_asd_vector,
                                                                  bins=8, density=True)

hist_dist_asd_control = go.Scattergl(x=bins_dist_asd_control_vector, y=hist_dist_asd_control_vector, name='Control',
                                     line={"color": "red"}, opacity=0.6, fill='tozeroy', fillcolor='red', mode='lines')

hist_dist_asd_asd = go.Scattergl(x=bins_dist_asd_asd_vector, y=hist_dist_asd_asd_vector, name='ASD', line={
    "color": "blue"}, fill='tozeroy', fillcolor='blue', opacity=0.5, mode='lines')

hist_idoa_asd_control = go.Scattergl(x=bins_idoa_asd_control_vector, y=hist_idoa_asd_control_vector, name='Control',
                                     line={"color": "red"}, opacity=0.6, fill='tozeroy', fillcolor='red', mode='lines')

hist_idoa_asd_asd = go.Scattergl(x=bins_idoa_asd_asd_vector, y=hist_idoa_asd_asd_vector, name='ASD', line={
    "color": "blue"}, fill='tozeroy', fillcolor='blue', opacity=0.5, mode='lines')

scatter_IDOA_asd = go.Scatter(x=idoa_control_asd_vector, y=idoa_asd_asd_vector, marker={"color": "blue"}, name='ASD',
                              mode="markers")

scatter_IDOA_control_asd = go.Scatter(x=idoa_control_control_vector_asd, y=idoa_asd_control_vector, marker={
    "color": "red"}, name='Control', mode="markers")

line_idoa_asd = go.Scattergl(x=[-4, 1], y=[-4, 1], line={"color": "black", "dash": 'dash'}, mode="lines",
                             showlegend=False)

scatter_dist_asd = go.Scatter(x=dist_control_asd_vector, y=dist_asd_asd_vector, marker={"color": "blue"}, name='ASD',
                              mode="markers")

scatter_dist_control_asd = go.Scatter(x=dist_control_control_vector_asd, y=dist_asd_control_vector, marker={
    "color": "red"}, name='Control', mode="markers")

line_dist_asd = go.Scattergl(x=[0.35, 0.7], y=[0.35, 0.7], line={"color": "black", "dash": 'dash'}, mode="lines",
                             showlegend=False)

scatter_asd = go.Scatter(x=dist_control_asd_vector, y=idoa_control_asd_vector, marker={"color": "blue"}, name='ASD',
                         mode="markers")

scatter_control_asd = go.Scatter(x=dist_control_control_vector_asd, y=idoa_control_control_vector_asd, marker={
    "color": "red"}, name='Control', mode="markers")

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        html.H1(children='IDOA and Distances methods - ASD')], className='header'),
        html.Div([
            dcc.Graph(
                id='Distance Histograms Control',
                figure={'data': [hist_dist_control_control, hist_dist_control_asd], 'layout': go.Layout(xaxis={
                        'title': {'text': 'Mean distance to control', 'font': {'size': 25}}, 'zeroline': False,
                        'showgrid': False, "showline": True, "linewidth": 2},
                        yaxis={'title': {"text": 'Frequency', 'font': {'size': 25}}, 'showgrid': False,
                               "showline": True, "linewidth": 2, "showticklabels": False}, xaxis_tickfont=dict(size=20),
                                yaxis_tickfont=dict(size=20), legend=dict(x=0.7, y=1, font=dict(size=17.5)), width=500,
                                height=500)})], style={'width': '45%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='IDOA Histograms Control',
                    figure={'data': [hist_idoa_control_control, hist_idoa_control_asd], 'layout': go.Layout(xaxis={
                            'title': {"text": 'IDOA w.r.t control', "font": {"size": 25}}, 'zeroline': False,
                            'showgrid': False, "showline": True, "linewidth": 2},
                            yaxis={'title': {"text": 'Frequency', 'font': {'size': 25}}, 'showgrid': False,
                            "showline": True, "linewidth": 2, "showticklabels": False}, xaxis_tickfont=dict(size=20),
                            yaxis_tickfont=dict(size=20), legend=dict(x=0, y=1, font=dict(size=17.5)), width=500,
                            height=500)})], style={'width': '45%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='Distance Histograms ASD',
                    figure={'data': [hist_dist_asd_control, hist_dist_asd_asd], 'layout': go.Layout(xaxis={'title': {
                            "text": 'Mean distance to ASD', "font": {"size": 25}}, 'zeroline': False, 'showgrid': False,
                            "showline": True, "linewidth": 2}, yaxis={'title': {"text": 'Frequency', 'font': {
                            'size': 25}}, 'showgrid': False, "showline": True, "linewidth": 2, "showticklabels": False},
                            xaxis_tickfont=dict(size=20), yaxis_tickfont=dict(size=20), legend=dict(x=0.7, y=1,
                            font=dict(size=17.5)), width=500, height=500)})], style={'width': '45%', 'display':
                    'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='IDOA Histograms ASD',
                    figure={'data': [hist_idoa_asd_control, hist_idoa_asd_asd], 'layout': go.Layout(xaxis={'title': {
                            "text": 'IDOA w.r.t ASD', "font": {"size": 25}}, 'zeroline': False, 'showgrid': False,
                            "showline": True, "linewidth": 2}, yaxis={'title': {"text": 'Frequency', 'font': {
                            'size': 25}}, 'showgrid': False, "showline": True, "linewidth": 2, "showticklabels": False},
                            xaxis_tickfont=dict(size=20), yaxis_tickfont=dict(size=20), legend=dict(x=0, y=1, font=dict(
                            size=17.5)), width=500, height=500)})], style={'width': '45%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='Combination IDOA - distances',
                    figure={'data': [scatter_control_asd, scatter_asd], 'layout': go.Layout(
                                     xaxis={"title": {"text": 'Mean distance to control', "font": {"size": 25}},
                                            'showgrid' : False, 'zeroline': False, "showline": True, "linewidth": 2,
                                            "range": [0.35, 0.7]},
                                     yaxis={"title": {"text": 'IDOA w.r.t control', "font": {"size": 25}},
                                            'showgrid' : False, 'zeroline': False, "showline": True, "linewidth": 2,
                                            "range": [-3, 1]}, xaxis_tickfont=dict(size=20), yaxis_tickfont=dict(
                                            size=20), width=500, height=500, legend=dict(x=0.7, y=0, font=dict(
                                            size=17.5)))})], style={'display': 'flex', 'flexDirection': 'row'}),
            html.Div([
                dcc.Graph(
                    id='IDOA - ASD',
                    figure={'data': [scatter_IDOA_control_asd, scatter_IDOA_asd, line_idoa_asd], 'layout': {'xaxis': {
                            'title': {"text": 'IDOA w.r.t control', "font": {"size": 25}}, "tickfont": {"size": 20},
                            'zeroline': False, 'scaleratio': 1, 'scaleanchor': 'y', 'showgrid': False, "showline": True,
                            "linewidth": 2}, 'yaxis': {'title': {"text": 'IDOA w.r.t ASD', "font": {"size": 25}},
                            'zeroline': False, "tickfont": {"size": 20}, 'scaleratio': 1, 'scaleanchor': 'x',
                            'showgrid': False, "showline": True, "linewidth": 2}, 'legend': {'x': 0, 'y': 1, "font": {
                            "size": 17.5}}, "width": 500, "height": 500}})], style={'width': '48%', 'display':
                            'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='Distances - ASD',
                    figure={'data': [scatter_dist_control_asd, scatter_dist_asd, line_dist_asd], 'layout': {'xaxis':
                            {'title': {"text": 'Mean distance to control', "font": {"size": 25}}, 'zeroline': False,
                            'scaleratio': 1, 'scaleanchor': 'y', 'showgrid': False, "showline": True,
                            "linewidth": 2, "tickfont": {"size": 20}},
                            'yaxis': {'title': {"text": 'Mean distance to ASD', "font": {"size": 25}},
                            'zeroline': False, "showline": True, "linewidth": 2, 'scaleratio': 1, 'scaleanchor': 'x',
                            'showgrid': False, "tickfont": {"size": 20}},
                            'legend': {'x': 0, 'y': 1, "font": {"size": 17.5}}, 'showgrid': False, "width": 500,
                            "height": 500,}},)], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='PCoA - ASD',
                    figure={'data': [PCoA_control_asd, PCoA_asd], 'layout': {'xaxis': {'title': {"text": 'PCoA1',
                            "font": {"size": 25}}, 'zeroline': False, 'scaleratio': 1, 'scaleanchor': 'y', 'showgrid':
                            False, "showline": True, "linewidth": 2, "tickfont": {"size": 20}},
                            'yaxis': {'title': {"text": 'PCoA2', "font": {"size": 25}}, "tickfont": {"size": 20},
                            'zeroline': False, 'scaleratio': 1, 'scaleanchor': 'x', 'showgrid': False, "showline": True,
                            "linewidth": 2}, 'legend' : {'x': 0, 'y': 1, "font": {"size": 17.5}}, "width": 500,
                            "height": 500}})], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='DOC - Control',
                    figure={'data': fig_control_asd, 'layout': go.Layout(xaxis={'title': {"text": 'Overlap', "font":
                            {"size": 25}}, 'range': [0.95, np.max(new_overlap_vector)], 'scaleratio':1,
                            'showgrid': False, "automargin": True, "showline": True, "linewidth": 2},
                            yaxis={'title': {"text": 'Dissimilarity', "font": {"size": 25}}, 'range': [0.25, 0.6],
                                   'scaleratio':1, 'showgrid': False, "automargin": True, "showline": True, "linewidth":
                                    2}, xaxis_tickfont=dict(size=20), yaxis_tickfont=dict(size=20), width=500,
                                    height=500)})], style={'width': '48%', 'display': 'inline-block'})])

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8080)
