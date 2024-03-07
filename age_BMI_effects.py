import pandas as pd
import numpy as np
from IDOA import IDOA
import plotly.graph_objects as go
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
import os
os.chdir(r'C:\Users\shaya\OneDrive\Desktop\IDOA')

# Normalization function.

def normalize_cohort(cohort):
    """
    cohort: numpy matrix, samples in rows
    """
    if cohort.ndim == 1:
        cohort_normalized = cohort / cohort.sum()
    else:
        cohort_normalized = cohort / np.linalg.norm(cohort, ord=1, axis=1, keepdims=True)
    return cohort_normalized

os.chdir(r'C:\Users\shaya\OneDrive\Desktop\IDOA\HMP_cohorts')

Stool = pd.read_excel('Stool.xlsx', header=None)

Stool_cohort = Stool.T
Stool_cohort = Stool_cohort.to_numpy()
Stool_cohort = normalize_cohort(Stool_cohort)
non_zero_columns = np.sum(Stool_cohort, axis=0) != 0
Stool_cohort = Stool_cohort[:, non_zero_columns]

def remove_low_mean_columns(arr):
    return arr[:, np.mean(arr, axis=0) >= 0.0001]
Stool_cohort = remove_low_mean_columns(Stool_cohort)

os.chdir(r'C:\Users\shaya\OneDrive\Desktop\IDOA')

Age_vals = pd.read_excel('Age_stool.xlsx', header=None)
Age_vals = Age_vals.values.T[0]
BMI_vals = pd.read_excel('BMI_stool.xlsx', header=None)
BMI_vals = BMI_vals.values[0]

def filter_matrix_by_vector(matrix, vector, value_range):
    """
    Filters the rows of a matrix based on a corresponding vector and a value range.

    Parameters:
        matrix (numpy.ndarray): A NumPy matrix with m rows and n columns.
        vector (numpy.ndarray): A NumPy vector with m elements.
        value_range (tuple): A tuple specifying the range of values (min_value, max_value).

    Returns:
        numpy.ndarray: A filtered matrix containing only the rows where the corresponding
                       values in the vector are within the specified range.
    """
    # Check if the dimensions are compatible
    if matrix.shape[0] != len(vector):
        raise ValueError("The number of rows in the matrix must match the length of the vector.")

    # Find the indices where the vector values are within the range
    min_value, max_value = value_range
    valid_indices = np.where((vector >= min_value) & (vector <= max_value))[0]

    # Filter the matrix using the valid indices
    filtered_matrix = matrix[valid_indices, :]

    return filtered_matrix

# Divide to age groups.
low_age_stool = filter_matrix_by_vector(Stool_cohort, Age_vals, (18, 24))
med_age_stool = filter_matrix_by_vector(Stool_cohort, Age_vals, (25, 31))
lar_age_stool = filter_matrix_by_vector(Stool_cohort, Age_vals, (32, 40))

# IDOA objects.
IDOA_object_low_low = IDOA(low_age_stool, low_age_stool, min_overlap=0.5, max_overlap=1, min_num_points=0,
                           percentage=50, method='percentage')
IDOA_object_low_med = IDOA(low_age_stool, med_age_stool, min_overlap=0.5, max_overlap=1, min_num_points=0,
                           percentage=50, method='percentage')
IDOA_object_low_lar = IDOA(low_age_stool, lar_age_stool, min_overlap=0.5, max_overlap=1, min_num_points=0,
                           percentage=50, method='percentage')
IDOA_object_med_low = IDOA(med_age_stool, low_age_stool, min_overlap=0.5, max_overlap=1, min_num_points=0,
                           percentage=50, method='percentage')
IDOA_object_med_med = IDOA(med_age_stool, med_age_stool, min_overlap=0.5, max_overlap=1, min_num_points=0,
                           percentage=50, method='percentage')
IDOA_object_med_lar = IDOA(med_age_stool, lar_age_stool, min_overlap=0.5, max_overlap=1, min_num_points=0,
                           percentage=50, method='percentage')
IDOA_object_lar_low = IDOA(lar_age_stool, low_age_stool, min_overlap=0.5, max_overlap=1, min_num_points=0,
                           percentage=50, method='percentage')
IDOA_object_lar_med = IDOA(lar_age_stool, med_age_stool, min_overlap=0.5, max_overlap=1, min_num_points=0,
                           percentage=50, method='percentage')
IDOA_object_lar_lar = IDOA(lar_age_stool, lar_age_stool, min_overlap=0.5, max_overlap=1, min_num_points=0,
                           percentage=50, method='percentage')

# IDOA values calculations.
IDOA_vector_low_low = IDOA_object_low_low.calc_idoa_vector()
IDOA_vector_low_med = IDOA_object_low_med.calc_idoa_vector()
IDOA_vector_low_lar = IDOA_object_low_lar.calc_idoa_vector()
IDOA_vector_med_low = IDOA_object_med_low.calc_idoa_vector()
IDOA_vector_med_med = IDOA_object_med_med.calc_idoa_vector()
IDOA_vector_med_lar = IDOA_object_med_lar.calc_idoa_vector()
IDOA_vector_lar_low = IDOA_object_lar_low.calc_idoa_vector()
IDOA_vector_lar_med = IDOA_object_lar_med.calc_idoa_vector()
IDOA_vector_lar_lar = IDOA_object_lar_lar.calc_idoa_vector()

# Divide to BMI groups.
low_bmi_stool = filter_matrix_by_vector(Stool_cohort, BMI_vals, (19, 26))
lar_bmi_stool = filter_matrix_by_vector(Stool_cohort, BMI_vals, (27, 34))

# IDOA objects.
IDOA_object_low_low_bmi = IDOA(low_bmi_stool, low_bmi_stool, min_overlap=0.5, max_overlap=1, min_num_points=0,
                               percentage=50, method='percentage')
IDOA_object_low_lar_bmi = IDOA(low_bmi_stool, lar_bmi_stool, min_overlap=0.5, max_overlap=1, min_num_points=0,
                               percentage=50, method='percentage')
IDOA_object_lar_low_bmi = IDOA(lar_bmi_stool, low_bmi_stool, min_overlap=0.5, max_overlap=1, min_num_points=0,
                               percentage=50, method='percentage')
IDOA_object_lar_lar_bmi = IDOA(lar_bmi_stool, lar_bmi_stool, min_overlap=0.5, max_overlap=1, min_num_points=0,
                               percentage=50, method='percentage')

# IDOA values calculations.
IDOA_vector_low_low_bmi = IDOA_object_low_low_bmi.calc_idoa_vector()
IDOA_vector_low_lar_bmi = IDOA_object_low_lar_bmi.calc_idoa_vector()
IDOA_vector_lar_low_bmi = IDOA_object_lar_low_bmi.calc_idoa_vector()
IDOA_vector_lar_lar_bmi = IDOA_object_lar_lar_bmi.calc_idoa_vector()

# Plots of histograms and PCoA

num_bins = 10

bin_size_low_low = (max(IDOA_vector_low_low) - min(IDOA_vector_low_low)) / num_bins
bin_size_low_med = (max(IDOA_vector_low_med) - min(IDOA_vector_low_med)) / num_bins
bin_size_low_lar = (max(IDOA_vector_low_lar) - min(IDOA_vector_low_lar)) / num_bins

hist_idoa_low_low = go.Histogram(
    x=IDOA_vector_low_low,
    xbins=dict(start=min(IDOA_vector_low_low), end=max(IDOA_vector_low_low), size=bin_size_low_low),
    opacity=0.5,
    marker=dict(color='blue'),
    name='Age from 18 to 24',
    histnorm='probability density'
)

hist_idoa_low_med = go.Histogram(
    x=IDOA_vector_low_med,
    xbins=dict(start=min(IDOA_vector_low_med), end=max(IDOA_vector_low_med), size=bin_size_low_med),
    opacity=0.5,
    marker=dict(color='red'),
    name='Age from 25 to 31',
    histnorm='probability density'
)

hist_idoa_low_lar = go.Histogram(
    x=IDOA_vector_low_lar,
    xbins=dict(start=min(IDOA_vector_low_lar), end=max(IDOA_vector_low_lar), size=bin_size_low_lar),
    opacity=0.5,
    marker=dict(color='grey'),
    name='Age from 32 to 40',
    histnorm='probability density'
)

layout = go.Layout(
    xaxis=dict(
        title='IDOA w.r.t Age from 18 to 24 cohort',
        linecolor='black',
        showline=True,
        zeroline=False,
        showgrid=False,
        titlefont=dict(family="Computer Modern", size=30),
        tickfont=dict(size=20)  
    ),
    yaxis=dict(
        title='Density',
        linecolor='black',
        showline=True,
        zeroline=False,
        showgrid=False,
        titlefont=dict(family="Computer Modern", size=30),
        tickfont=dict(size=20)  
    ),
    barmode='overlay',
    legend=dict(x=0, y=1, font=dict(size=25, family="Computer Modern")),
    width=900,
    height=900,
    plot_bgcolor='white'
)

fig = go.Figure(data=[hist_idoa_low_low, hist_idoa_low_med, hist_idoa_low_lar], layout=layout)
fig.show()

num_bins = 10

bin_size_med_low = (max(IDOA_vector_med_low) - min(IDOA_vector_med_low)) / num_bins
bin_size_med_med = (max(IDOA_vector_med_med) - min(IDOA_vector_med_med)) / num_bins
bin_size_med_lar = (max(IDOA_vector_med_lar) - min(IDOA_vector_med_lar)) / num_bins

histogram_trace_med_low = go.Histogram(
    x=IDOA_vector_med_low,
    xbins=dict(start=min(IDOA_vector_med_low), end=max(IDOA_vector_med_low), size=bin_size_med_low),
    opacity=0.5,
    marker=dict(color='blue'),
    name='Age from 18 to 24',
    histnorm='probability density'
)

histogram_trace_med_med = go.Histogram(
    x=IDOA_vector_med_med,
    xbins=dict(start=min(IDOA_vector_med_med), end=max(IDOA_vector_med_med), size=bin_size_med_med),
    opacity=0.5,
    marker=dict(color='red'),
    name='Age from 25 to 31',
    histnorm='probability density'
)

histogram_trace_med_lar = go.Histogram(
    x=IDOA_vector_med_lar,
    xbins=dict(start=min(IDOA_vector_med_lar), end=max(IDOA_vector_med_lar), size=bin_size_med_lar),
    opacity=0.5,
    marker=dict(color='grey'),
    name='Age from 32 to 40',
    histnorm='probability density'
)

layout = go.Layout(
    xaxis=dict(
        title='IDOA w.r.t Age from 25 to 31 cohort',
        linecolor='black',
        showline=True,
        zeroline=False,
        showgrid=False,
        titlefont=dict(family="Computer Modern", size=30),
        tickfont=dict(size=20) 
    ),
    yaxis=dict(
        title='Density',
        linecolor='black',
        showline=True,
        zeroline=False,
        showgrid=False,
        titlefont=dict(family="Computer Modern", size=30),
        tickfont=dict(size=20)  
    ),
    barmode='overlay',
    legend=dict(x=0, y=1, font=dict(size=25, family="Computer Modern")),
    width=900,
    height=900,
    plot_bgcolor='white'
)

fig = go.Figure(data=[histogram_trace_med_low, histogram_trace_med_med,
                      histogram_trace_med_lar], layout=layout)
fig.show()

num_bins = 10

bin_size_lar_low = (max(IDOA_vector_lar_low) - min(IDOA_vector_lar_low)) / num_bins
bin_size_lar_med = (max(IDOA_vector_lar_med) - min(IDOA_vector_lar_med)) / num_bins
bin_size_lar_lar = (max(IDOA_vector_lar_lar) - min(IDOA_vector_lar_lar)) / num_bins

histogram_trace_lar_low = go.Histogram(
    x=IDOA_vector_lar_low,
    xbins=dict(start=min(IDOA_vector_lar_low), end=max(IDOA_vector_lar_low), size=bin_size_lar_low),
    opacity=0.5,
    marker=dict(color='blue'),
    name='Age from 18 to 24',
    histnorm='probability density'
)

histogram_trace_lar_med = go.Histogram(
    x=IDOA_vector_lar_med,
    xbins=dict(start=min(IDOA_vector_lar_med), end=max(IDOA_vector_lar_med), size=bin_size_lar_med),
    opacity=0.5,
    marker=dict(color='red'),
    name='Age from 25 to 31',
    histnorm='probability density'
)

histogram_trace_lar_lar = go.Histogram(
    x=IDOA_vector_lar_lar,
    xbins=dict(start=min(IDOA_vector_lar_lar), end=max(IDOA_vector_lar_lar), size=bin_size_lar_lar),
    opacity=0.5,
    marker=dict(color='grey'),
    name='Age from 32 to 40',
    histnorm='probability density'
)

layout = go.Layout(
    xaxis=dict(
        title='IDOA w.r.t Age from 32 to 40 cohort',
        linecolor='black',
        showline=True,
        zeroline=False,
        showgrid=False,
        titlefont=dict(family="Computer Modern", size=30),
        tickfont=dict(size=20)
    ),
    yaxis=dict(
        title='Density',
        linecolor='black',
        showline=True,
        zeroline=False,
        showgrid=False,
        titlefont=dict(family="Computer Modern", size=30),
        tickfont=dict(size=20) 
    ),
    barmode='overlay',
    legend=dict(x=0, y=1, font=dict(size=25, family="Computer Modern")),
    width=900,
    height=900,
    plot_bgcolor='white'
)

fig = go.Figure(data=[histogram_trace_lar_low, histogram_trace_lar_med,
                      histogram_trace_lar_lar], layout=layout)
fig.show()

num_bins = 10

bin_size_low_low = (max(IDOA_vector_low_low_bmi) - min(IDOA_vector_low_low_bmi)) / num_bins
bin_size_low_lar = (max(IDOA_vector_low_lar_bmi) - min(IDOA_vector_low_lar_bmi)) / num_bins

histogram_trace_low_low = go.Histogram(
    x=IDOA_vector_low_low_bmi,
    xbins=dict(start=min(IDOA_vector_low_low_bmi), end=max(IDOA_vector_low_low_bmi),
               size=bin_size_low_low),
    opacity=0.5,
    marker=dict(color='blue'),
    name='BMI from 19 to 26',
    histnorm='probability density'
)

histogram_trace_low_lar = go.Histogram(
    x=IDOA_vector_low_lar_bmi,
    xbins=dict(start=min(IDOA_vector_low_lar_bmi), end=max(IDOA_vector_low_lar_bmi),
               size=bin_size_low_lar),
    opacity=0.5,
    marker=dict(color='red'),
    name='BMI from 27 to 34',
    histnorm='probability density'
)

layout = go.Layout(
    xaxis=dict(
        title='IDOA w.r.t BMI from 19 to 26 cohort',
        linecolor='black',
        showline=True,
        zeroline=False,
        showgrid=False,
        titlefont=dict(family="Computer Modern", size=30),
        tickfont=dict(size=20)
    ),
    yaxis=dict(
        title='Density',
        linecolor='black',
        showline=True,
        zeroline=False,
        showgrid=False,
        titlefont=dict(family="Computer Modern", size=30),
        tickfont=dict(size=20) 
    ),
    barmode='overlay',
    legend=dict(x=0, y=1, font=dict(size=25, family="Computer Modern")),
    width=900,
    height=900,
    plot_bgcolor='white'
)

fig = go.Figure(data=[histogram_trace_low_low, histogram_trace_low_lar], layout=layout)
fig.show()

num_bins = 10

bin_size_lar_low = (max(IDOA_vector_lar_low_bmi) - min(IDOA_vector_lar_low_bmi)) / num_bins
bin_size_lar_lar = (max(IDOA_vector_lar_lar_bmi) - min(IDOA_vector_lar_lar_bmi)) / num_bins

histogram_trace_lar_low = go.Histogram(
    x=IDOA_vector_lar_low_bmi,
    xbins=dict(start=min(IDOA_vector_lar_low_bmi), end=max(IDOA_vector_lar_low_bmi),
               size=bin_size_lar_low),
    opacity=0.5,
    marker=dict(color='blue'),
    name='BMI from 19 to 26',
    histnorm='probability density'
)

histogram_trace_lar_lar = go.Histogram(
    x=IDOA_vector_lar_lar_bmi,
    xbins=dict(start=min(IDOA_vector_lar_lar_bmi), end=max(IDOA_vector_lar_lar_bmi),
               size=bin_size_lar_lar),
    opacity=0.5,
    marker=dict(color='red'),
    name='BMI from 27 to 34',
    histnorm='probability density'
)

layout = go.Layout(
    xaxis=dict(
        title='IDOA w.r.t BMI from 27 to 34 cohort',
        linecolor='black',
        showline=True,
        zeroline=False,
        showgrid=False,
        titlefont=dict(family="Computer Modern", size=30),
        tickfont=dict(size=20) 
    ),
    yaxis=dict(
        title='Density',
        linecolor='black',
        showline=True,
        zeroline=False,
        showgrid=False,
        titlefont=dict(family="Computer Modern", size=30),
        tickfont=dict(size=20)  
    ),
    barmode='overlay',
    legend=dict(x=0, y=1, font=dict(size=25, family="Computer Modern")),
    width=900,
    height=900,
    plot_bgcolor='white'
)

fig = go.Figure(data=[histogram_trace_lar_low, histogram_trace_lar_lar], layout=layout)
fig.show()

combined_data = np.vstack([low_bmi_stool, lar_bmi_stool])

# Compute dissimilarity
dissimilarity = pairwise_distances(combined_data, metric='braycurtis')

# Perform MDS
embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_result = embedding.fit_transform(dissimilarity)

# Create DataFrame
df = pd.DataFrame(mds_result, columns=['PCoA 1', 'PCoA 2'])

# Assign colors
point_colors = ['red'] * 190
point_colors[0:142] = ['blue'] * 142
df['color'] = point_colors

# Create Plotly figure
fig_pcoa = go.Figure()

# Add traces for each color group and specify the label
fig_pcoa.add_trace(go.Scatter(x=df[df['color'] == 'blue']['PCoA 1'], 
                              y=df[df['color'] == 'blue']['PCoA 2'],
                              mode='markers',
                              marker=dict(color='blue'),
                              name='BMI from 19 to 26'))  

fig_pcoa.add_trace(go.Scatter(x=df[df['color'] == 'red']['PCoA 1'], 
                              y=df[df['color'] == 'red']['PCoA 2'],
                              mode='markers',
                              marker=dict(color='red'),
                              name='BMI from 27 to 34'))  

# Update layout
fig_pcoa.update_layout(
    xaxis=dict(
        linecolor='black',
        showline=True,
        showgrid=False,
        showticklabels=True,
        zeroline=False,
        title_text='PCo1'
    ),
    yaxis=dict(
        linecolor='black',
        showline=True,
        showgrid=False,
        showticklabels=True,
        zeroline=False,
        title_text='PCo2'
    ),
    font=dict(
        family="latex",
        size=30,
        color="black"
    ),
    width=800,
    height=800,
    plot_bgcolor='white',
    showlegend=True, 
    legend=dict(
        x=0.05,
        y=1,  
        traceorder="normal",
        orientation="v",
        font=dict(
            family="latex",
            size=25,
            color="black"
        )
    )
)

# Show figure
fig_pcoa.show()

combined_data = np.vstack([low_age_stool, med_age_stool, lar_age_stool])

# Compute dissimilarity
dissimilarity = pairwise_distances(combined_data, metric='braycurtis')

# Perform MDS
embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_result = embedding.fit_transform(dissimilarity)

# Create DataFrame
df = pd.DataFrame(mds_result, columns=['PCoA 1', 'PCoA 2'])

# Assign colors
point_colors = ['black'] * 190
point_colors[0:69] = ['blue'] * 69
point_colors[69:157] = ['red'] * 88
df['color'] = point_colors

# Create Plotly figure
fig_pcoa = go.Figure()

# Add traces for each color group and specify the label
fig_pcoa.add_trace(go.Scatter(x=df[df['color'] == 'blue']['PCoA 1'], 
                              y=df[df['color'] == 'blue']['PCoA 2'],
                              mode='markers',
                              marker=dict(color='blue'),
                              name='Age from 18 to 24'))  

fig_pcoa.add_trace(go.Scatter(x=df[df['color'] == 'red']['PCoA 1'], 
                              y=df[df['color'] == 'red']['PCoA 2'],
                              mode='markers',
                              marker=dict(color='red'),
                              name='Age from 25 to 31'))  

fig_pcoa.add_trace(go.Scatter(x=df[df['color'] == 'black']['PCoA 1'], 
                              y=df[df['color'] == 'black']['PCoA 2'],
                              mode='markers',
                              marker=dict(color='black'),
                              name='Age from 32 to 40'))  

# Update layout
fig_pcoa.update_layout(
    xaxis=dict(
        linecolor='black',
        showline=True,
        showgrid=False,
        showticklabels=True,
        zeroline=False,
        title_text='PCo1'
    ),
    yaxis=dict(
        linecolor='black',
        showline=True,
        showgrid=False,
        showticklabels=True,
        zeroline=False,
        title_text='PCo2'
    ),
    font=dict(
        family="latex",
        size=30,
        color="black"
    ),
    width=800,
    height=800,
    plot_bgcolor='white',
    showlegend=True, 
    legend=dict(
        x=0.05,
        y=1,  
        traceorder="normal",
        orientation="v",
        font=dict(
            family="latex",
            size=25,
            color="black"
        )
    )
)

fig_pcoa.show()