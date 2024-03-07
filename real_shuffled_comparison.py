import os
os.chdir(r'C:\Users\shaya\OneDrive\Desktop\IDOA_project')
from IDOA import IDOA
from DOC import DOC
from Shuffle_cohort import ShuffledCohort
import pandas as pd
import numpy as np
from Functions import calc_bray_curtis_dissimilarity

import os
os.chdir(r'C:\Users\shaya\OneDrive\Desktop\IDOA\HMP_cohorts')

real = pd.read_excel('Right retroauricular crease.xlsx', header=None)

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


# Filtering and normalization, the cohort is normalized so the samples (rows) sum up to one.
# Species that are non-zero at only one sample or less and species with less than 0.01% of the total abundance are
# filtered.

real_cohort = real.values
real_cohort = real.T
real_cohort = real_cohort.to_numpy()
real_cohort = normalize_cohort(real_cohort)
non_zero_columns = np.sum(real_cohort, axis=0) != 0
# remove zero columns
real_cohort = real_cohort[:, non_zero_columns]
def remove_low_mean_columns(mat, threshold=0.0001):
    return mat[:, np.mean(mat, axis=0) >= threshold]
def filter_out_single_nonzero_columns(matrix):
    # Count the number of nonzero elements in each column
    nonzero_count = np.count_nonzero(matrix, axis=0)
    
    # Find the columns where the count is not equal to 1
    not_single_nonzero_columns = np.where(nonzero_count != 1)[0]
    
    # Select only those columns
    filtered_matrix = matrix[:, not_single_nonzero_columns]
    
    return filtered_matrix
# remove the species with less than 0.01% of the total abundance
real_cohort = remove_low_mean_columns(real_cohort)
# remove the species that non-zero only at one sample
real_cohort = filter_out_single_nonzero_columns(real_cohort)

real_cohort = normalize_cohort(real_cohort)

# Plot DOC function, the function plots the DOC and fits a LOWESS curve to the scatterplot.
import plotly.graph_objects as go
import statsmodels.api as sm

def scatterplot_plotly(x, y, xlabel="Overlap", ylabel="Dissimilarity", title="DOC", size=2,
                       frac=0.1, x_lower_limit=None, x_upper_limit=None):
    """
    x: Overlap values.
    y: Dissimilarity values.
    xlabel, ylabel: labels to the plot.
    size: the size of the points on the graph.
    x_lower_limit, x_upper_limit: optional minimal and maximal values for the Overlap axis. 
    """
    
    # Fit the LOWESS curve
    lowess_result = sm.nonparametric.lowess(y, x, frac=frac)
    lowess_x = lowess_result[:, 0]
    lowess_y = lowess_result[:, 1]
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add the scatter plot data
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data', marker=dict(size=size)))
    
    # Add the LOWESS curve
    fig.add_trace(go.Scatter(x=lowess_x, y=lowess_y, mode='lines', name='LOWESS',
                             line=dict(color='red', width=3)))
    
    # Layout
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis={
            'showgrid': False,
            'zeroline': False,  
            'linecolor': 'white', 
            'linewidth': 1,  
            'title_font': {
                'size': 30,
                'family': "latex"
            },
            'range': [x_lower_limit,
                      x_upper_limit] if x_lower_limit is not None and x_upper_limit is not None else None
        },
        yaxis={
            'showgrid': False,
            'zeroline': False,  
            'linecolor': 'white', 
            'linewidth': 1,  
            'title_font': {
                'size': 30,
                'family': "latex"
            }
        },
        template="plotly_dark",
        width=700,
        height=700,
        showlegend=False
    )
    
    fig.show()

# Create shuffled cohort using the ShuffledCohort class.

shuffled_real_cohort = ShuffledCohort(real_cohort).create_shuffled_cohort()

# Calculation of the IDOA values of samples in the real and shuffled cohort with respect to the real cohort using the
# IDOA class, the threshold for the IDOA calculation is top 50% of the highest overlap.

IDOA_object_real = IDOA(real_cohort, real_cohort, min_overlap=0.5, max_overlap=1, min_num_points=0, percentage=50)
IDOA_real_vector = IDOA_object_real.calc_idoa_vector()

IDOA_object_shuffled = IDOA(real_cohort, shuffled_real_cohort, min_overlap=0.5, max_overlap=1, min_num_points=0,
                            percentage=50)
IDOA_shuffled_vector = IDOA_object_shuffled.calc_idoa_vector()

# Plot hte IDOA histograms.

#bin_size = 0.05
num_bins = 30

bin_size = (max(max(IDOA_real_vector), max(
    IDOA_shuffled_vector)) - min(min(IDOA_real_vector), min(
    IDOA_shuffled_vector))) / num_bins

# histogram for the IDOA values for the real cohort.
histogram_trace_real = go.Histogram(
    x=IDOA_real_vector,
    xbins=dict(start=min(IDOA_real_vector), end=max(IDOA_real_vector), size=bin_size),
    opacity=0.5,
    marker=dict(color='blue'),
    name='Real',
    histnorm='probability density'
)

# histogram for the IDOA values for the shuffled cohort.
histogram_trace_shuffled = go.Histogram(
    x=IDOA_shuffled_vector,
    xbins=dict(start=min(IDOA_shuffled_vector), end=max(IDOA_shuffled_vector), size=bin_size),
    opacity=0.5,
    marker=dict(color='red'),
    name='Shuffled',
    histnorm='probability density'
)

# Layout
layout = go.Layout(
    xaxis=dict(
        title='IDOA',
        zeroline=False,  # Hide default x-axis line
        showline=True,  # Show custom x-axis line
        linecolor='black',  # Set color of x-axis line to black
        showgrid=False,
        titlefont=dict(family="Computer Modern", size=30),
        tickfont=dict(size=20)
    ),
    yaxis=dict(
        title='Density',
        zeroline=False,  # Hide default y-axis line
        showline=True,  # Show custom y-axis line
        linecolor='black',  # Set color of y-axis line to black
        showgrid=False,
        titlefont=dict(family="Computer Modern", size=30),
        tickfont=dict(size=20)
    ),
    barmode='overlay',
    legend=dict(x=0, y=1, font=dict(size=25, family="Computer Modern")),
    width=600,
    height=600,
    plot_bgcolor='white'  # Set background color to white
)

fig = go.Figure(data=[histogram_trace_real, histogram_trace_shuffled], layout=layout)
fig.show()

from scipy.spatial.distance import braycurtis

# Calculate the mean distance for the real and shuffled samples from the real cohort samples.

distance_real_vector = calc_bray_curtis_dissimilarity(real_cohort, real_cohort)
distance_shuffled_vector = calc_bray_curtis_dissimilarity(shuffled_real_cohort, real_cohort)

# Plot hte Mean distance histograms.

num_bins = 30

bin_size = (max(max(distance_real_vector), max(
    distance_shuffled_vector)) - min(min(distance_real_vector), min(
    distance_shuffled_vector))) / num_bins

# histogram for the mean distance values for the real cohort.
histogram_trace_real = go.Histogram(
    x=distance_real_vector,
    xbins=dict(start=min(distance_real_vector), end=max(distance_real_vector), size=bin_size),
    opacity=0.5,
    marker=dict(color='blue'),
    name='Real',
    histnorm='probability density'
)

# histogram for the mean distance values for the shuffled cohort.
histogram_trace_shuffled = go.Histogram(
    x=distance_shuffled_vector,
    xbins=dict(start=min(distance_shuffled_vector), end=max(distance_shuffled_vector), size=bin_size),
    opacity=0.5,
    marker=dict(color='red'),
    name='Shuffled',
    histnorm='probability density'
)

# Layout
layout = go.Layout(
    xaxis=dict(
        title='Mean Bray Curtis',
        zeroline=False,  # Hide default x-axis line
        showline=True,  # Show custom x-axis line
        linecolor='black',  # Set color of x-axis line to black
        showgrid=False,
        titlefont=dict(family="Computer Modern", size=30),
        tickfont=dict(size=20)
    ),
    yaxis=dict(
        title='Density',
        zeroline=False,  # Hide default y-axis line
        showline=True,  # Show custom y-axis line
        linecolor='black',  # Set color of y-axis line to black
        showgrid=False,
        titlefont=dict(family="Computer Modern", size=30),
        tickfont=dict(size=20)
    ),
    barmode='overlay',
    legend=dict(x=0.7, y=1, font=dict(size=25, family="Computer Modern")),
    width=600,
    height=600,
    plot_bgcolor='white'  # Set background color to white
)

fig = go.Figure(data=[histogram_trace_real, histogram_trace_shuffled], layout=layout)
fig.show()