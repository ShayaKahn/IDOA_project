import os
os.chdir(r'C:\Users\shaya\OneDrive\Desktop\IDOA_project')
from IDOA import IDOA
from DOC import DOC
from Shuffle_cohort import ShuffledCohort
import pandas as pd
import numpy as np

# Function that organizes a SparseDOSSA2 cohort.

def create_fitted_cohort(cohort, binary_vector):
    """
    cohort: numpy array of shape (n_features, n_samples), represent the otu matrix.
    binary_vector: numpy array of shape (n_features,), represent logical the binary vector where the indexes of the
                   non-filtered species is 1 else 0.
    """
    fitted_cohort = np.zeros((len(binary_vector), cohort.shape[1]))

    fitted_cohort[binary_vector == 1, :] = cohort

    return fitted_cohort

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

# Filtering and normalization, the cohort is normalized so the samples (rows) sum up to one.
# Species that are non-zero at only one sample or less and species with less than 0.05% of the total abundance are
# filtered.

Stool = pd.read_excel('Stool.xlsx', header=None)
Stool_cohort = Stool.values
Stool_cohort = Stool.T
Stool_cohort = Stool_cohort.to_numpy()
Stool_cohort = normalize_cohort(Stool_cohort)
non_zero_columns = np.sum(Stool_cohort, axis=0) != 0
Stool_cohort = Stool_cohort[:, non_zero_columns]
def remove_low_mean_columns(arr):
    return arr[:, np.mean(arr, axis=0) >= 0.0005]
Stool_cohort = remove_low_mean_columns(Stool_cohort)
Stool_cohort = normalize_cohort(Stool_cohort)

# Creating a shuffled cohort using ShuffledCohort class.

stool_cohort_shuffled_object = ShuffledCohort(Stool_cohort)
stool_cohort_shuffled = stool_cohort_shuffled_object.create_shuffled_cohort()

os.chdir(r'C:\Users\shaya\OneDrive\Desktop\IDOA')

# Load MIDAS stool cohort.

MIDAS_stool = pd.read_excel('Midas_stool.xlsx', header=None)
MIDAS_stool_cohort = MIDAS_stool.values

os.chdir(r'C:\Users\shaya\OneDrive\Desktop\IDOA')

# Load SparseDOSSA2 stool cohort and filtered species binary vector.

SparseDOSSA2_stool = pd.read_excel('SparseDOSSA2_simulated_samples_stool_new.xlsx', header=None)
SparseDOSSA2_stool = SparseDOSSA2_stool.values
SparseDOSSA2_stool_filtered_species = pd.read_excel('SparseDOSSA2_filtered_species_binary_stool_new.xlsx',
                                                    header=None)
SparseDOSSA2_stool_filtered_species = SparseDOSSA2_stool_filtered_species.values
SparseDOSSA2_stool_filtered_species = SparseDOSSA2_stool_filtered_species.flatten()

SparseDOSSA2_stool_cohort = create_fitted_cohort(SparseDOSSA2_stool, 
                                                 SparseDOSSA2_stool_filtered_species)

SparseDOSSA2_stool_cohort = SparseDOSSA2_stool_cohort.T

# Set the SparseDOSSA2 cohort to be the same size as the Stool_cohort.
SparseDOSSA2_stool_cohort_part = SparseDOSSA2_stool_cohort[0:Stool_cohort.shape[0], :]

# Calculate the DOC for the different cohorts.

DOC_Stool = DOC(Stool_cohort)
doc_mat_Stool = DOC_Stool.calc_doc()
o_stool = doc_mat_Stool[0, :]
d_stool = doc_mat_Stool[1, :]

DOC_Stool_MIDAS = DOC(MIDAS_stool_cohort)
doc_mat_Stool_MIDAS = DOC_Stool_MIDAS.calc_doc()
o_stool_midas = doc_mat_Stool_MIDAS[0, :]
d_stool_midas = doc_mat_Stool_MIDAS[1, :]

DOC_Stool_SparseDOSSA2 = DOC(SparseDOSSA2_stool_cohort_part)
doc_mat_Stool_SparseDOSSA2 = DOC_Stool_SparseDOSSA2.calc_doc()
o_stool_sparsedossa2 = doc_mat_Stool_SparseDOSSA2[0, :]
d_stool_sparsedossa2 = doc_mat_Stool_SparseDOSSA2[1, :]

DOC_shuffled = DOC(stool_cohort_shuffled)
doc_mat_Stool_shuffled = DOC_shuffled.calc_doc()
o_stool_shuffled = doc_mat_Stool_shuffled[0, :]
d_stool_shuffled = doc_mat_Stool_shuffled[1, :]

import plotly.graph_objects as go
import statsmodels.api as sm

# Plot the DOC for the different cohorts.

def scatterplot_plotly(x, y, xlabel="Overlap", ylabel="Dissimilarity", title="DOC", size=2,
                       frac=0.1, x_lower_limit=None, x_upper_limit=None):
    # Fit the LOWESS curve
    lowess_result = sm.nonparametric.lowess(y, x, frac=frac)
    lowess_x = lowess_result[:, 0]
    lowess_y = lowess_result[:, 1]
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add the scatter plot data
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data', marker=dict(size=size)))
    
    # Add the LOWESS curve with increased width
    fig.add_trace(go.Scatter(x=lowess_x, y=lowess_y, mode='lines', name='LOWESS',
                             line=dict(color='red', width=3)))
    
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

scatterplot_plotly(o_stool, d_stool, x_lower_limit=0, x_upper_limit=1)
scatterplot_plotly(o_stool_midas, d_stool_midas, x_lower_limit=0, x_upper_limit=1)
scatterplot_plotly(o_stool_sparsedossa2, d_stool_sparsedossa2, x_lower_limit=0, x_upper_limit=1)
scatterplot_plotly(o_stool_shuffled, d_stool_shuffled, x_lower_limit=0, x_upper_limit=1)

# Plot the histograms of the Dissimilarity for the different cohorts.
num_bins = 10

bin_size = (max(max(d_stool), max(d_stool_shuffled)) - min(min(d_stool), min(
    d_stool_shuffled))) / num_bins

histogram_trace_real = go.Histogram(
    x=d_stool,
    xbins=dict(start=min(d_stool), end=max(d_stool), size=bin_size),
    opacity=0.5,
    marker=dict(color='blue'),
    name='Real',
    histnorm='probability density'
)

histogram_trace_MIDAS = go.Histogram(
    x=d_stool_midas,
    xbins=dict(start=min(d_stool_midas), end=max(d_stool_midas), size=bin_size),
    opacity=0.5,
    marker=dict(color='chartreuse'),
    name='MIDAS',
    histnorm='probability density'
)

histogram_trace_SparseDOSSA2 = go.Histogram(
    x=d_stool_sparsedossa2,
    xbins=dict(start=min(d_stool_sparsedossa2), end=max(d_stool_sparsedossa2), size=bin_size),
    opacity=0.5,
    marker=dict(color='red'),
    name='SparseDOSSA2',
    histnorm='probability density'
)

histogram_trace_shuffled = go.Histogram(
    x=d_stool_shuffled,
    xbins=dict(start=min(d_stool_shuffled), end=max(d_stool_shuffled), size=bin_size),
    opacity=0.5,
    marker=dict(color='grey'),
    name='shuffled',
    histnorm='probability density'
)

layout = go.Layout(
    xaxis={
        'title': {"text": 'Dissimilarity', "font": {"size": 30, "family": "Computer Modern"}},
        'zeroline': False,
        'showgrid': False,
        "showline": True,
        "linewidth": 2,
        'tickfont': dict(size=20),
    },
    yaxis={
        'title': {"text": 'Density', 'font': {'size': 30, "family": "Computer Modern"}},
        'showgrid': False,
        "showline": True,
        "linewidth": 2,
        'tickfont': dict(size=20)
    },
    legend=dict(x=0, y=1, font=dict(size=25, family="latex")),
    width=700,
    height=700,
    barmode='overlay',
)

fig = go.Figure(data=[histogram_trace_real, histogram_trace_MIDAS, histogram_trace_SparseDOSSA2,
                      histogram_trace_shuffled], layout=layout)

fig.show()

# Plot the histograms of the Overlap for the different cohorts.

num_bins = 10

bin_size = (max(max(o_stool), max(o_stool_shuffled)) - min(min(o_stool), min(
    o_stool_shuffled))) / num_bins

histogram_trace_real = go.Histogram(
    x=o_stool,
    xbins=dict(start=min(o_stool), end=max(o_stool), size=bin_size),
    opacity=0.5,
    marker=dict(color='blue'),
    name='Real',
    histnorm='probability density'
)

histogram_trace_MIDAS = go.Histogram(
    x=o_stool_midas,
    xbins=dict(start=min(o_stool_midas), end=max(o_stool_midas), size=bin_size),
    opacity=0.5,
    marker=dict(color='chartreuse'),
    name='MIDAS',
    histnorm='probability density'
)

histogram_trace_SparseDOSSA2 = go.Histogram(
    x=o_stool_sparsedossa2,
    xbins=dict(start=min(o_stool_sparsedossa2), end=max(o_stool_sparsedossa2), size=bin_size),
    opacity=0.5,
    marker=dict(color='red'),
    name='SparseDOSSA2',
    histnorm='probability density'
)

histogram_trace_shuffled = go.Histogram(
    x=o_stool_shuffled,
    xbins=dict(start=min(o_stool_shuffled), end=max(o_stool_shuffled), size=bin_size),
    opacity=0.5,
    marker=dict(color='grey'),
    name='shuffled',
    histnorm='probability density'
)

layout = go.Layout(
    xaxis={
        'title': {"text": 'Overlap', "font": {"size": 30, "family": "Computer Modern"}},
        'zeroline': False,
        'showgrid': False,
        "showline": True,
        "linewidth": 2,
        'tickfont': dict(size=20),
    },
    yaxis={
        'title': {"text": 'Density', 'font': {'size': 30, "family": "Computer Modern"}},
        'showgrid': False,
        "showline": True,
        "linewidth": 2,
        'tickfont': dict(size=20)
    },
    legend=dict(x=0, y=1, font=dict(size=25, family="latex")),
    width=700,
    height=700,
    barmode='overlay',
)

fig = go.Figure(data=[histogram_trace_real, histogram_trace_MIDAS, histogram_trace_SparseDOSSA2,
                      histogram_trace_shuffled], layout=layout)

fig.show()

from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

# Plot PCoA of all groups

MIDAS_stool_cohort = MIDAS_stool.values

combined_data = np.vstack([Stool_cohort, MIDAS_stool_cohort, SparseDOSSA2_stool_cohort_part, 
                           stool_cohort_shuffled])

# Compute dissimilarity
dissimilarity = pairwise_distances(combined_data, metric='braycurtis')

# Perform MDS
embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_result = embedding.fit_transform(dissimilarity)

# Create DataFrame
df = pd.DataFrame(mds_result, columns=['PCoA 1', 'PCoA 2'])

# Assign colors
point_colors = ['grey'] * Stool_cohort.shape[0] * 4
point_colors[0:Stool_cohort.shape[0]] = ['blue'] * Stool_cohort.shape[0]
point_colors[Stool_cohort.shape[0]:Stool_cohort.shape[0] * 2] = ['chartreuse'] * Stool_cohort.shape[0]
point_colors[Stool_cohort.shape[0] * 2:Stool_cohort.shape[0] * 3] = ['red'] * Stool_cohort.shape[0]
df['color'] = point_colors

# Create Plotly figure
fig_pcoa = go.Figure()

# Add traces for each color group and specify the label
fig_pcoa.add_trace(go.Scatter(x=df[df['color'] == 'blue']['PCoA 1'], 
                              y=df[df['color'] == 'blue']['PCoA 2'],
                              mode='markers',
                              marker=dict(color='blue'),
                              name='Real'))  

fig_pcoa.add_trace(go.Scatter(x=df[df['color'] == 'chartreuse']['PCoA 1'], 
                              y=df[df['color'] == 'chartreuse']['PCoA 2'],
                              mode='markers',
                              marker=dict(color='chartreuse'),
                              name='MIDAS'))  

fig_pcoa.add_trace(go.Scatter(x=df[df['color'] == 'red']['PCoA 1'], 
                              y=df[df['color'] == 'red']['PCoA 2'],
                              mode='markers',
                              marker=dict(color='red'),
                              name='SparseDOSSA2'))  

fig_pcoa.add_trace(go.Scatter(x=df[df['color'] == 'grey']['PCoA 1'], 
                              y=df[df['color'] == 'grey']['PCoA 2'],
                              mode='markers',
                              marker=dict(color='grey'),
                              name='Shuffled')) 

# Update layout
fig_pcoa.update_layout(
    xaxis=dict(
        showline=False,
        showgrid=False,
        showticklabels=True,
        zeroline=False,
        title_text='PCoA 1',
        title_font=dict(  
            size=30
        ),
        tickfont=dict(  
            size=18
        )
    ),
    yaxis=dict(
        showline=False,
        showgrid=False,
        showticklabels=True,
        zeroline=False,
        title_text='PCoA 2',
        title_font=dict( 
            size=30
        ),
        tickfont=dict(  
            size=18
        )
    ),
    font=dict(
        family="latex",
        size=18,
        color="black"
    ),
    width=700,
    height=700,
    showlegend=True,
    legend=dict(
        x=0,
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

# Clculate IDOA for all groups

IDOA_object_real = IDOA(Stool_cohort, Stool_cohort, min_overlap=0.9,
                        max_overlap=1, zero_overlap=0, identical=True, min_num_points=0,
                        percentage=50, method='percentage')
IDOA_vector_real = IDOA_object_real.calc_idoa_vector()
IDOA_object_MIDAS = IDOA(MIDAS_stool_cohort, MIDAS_stool_cohort,
                         min_overlap=0.9, max_overlap=1, zero_overlap=0,
                         identical=True, min_num_points=0, percentage=50, method='percentage')
IDOA_vector_MIDAS = IDOA_object_MIDAS.calc_idoa_vector()
IDOA_object_shuffled = IDOA(stool_cohort_shuffled, stool_cohort_shuffled,
                            min_overlap=0.9, max_overlap=1, zero_overlap=0,
                            identical=True, min_num_points=0, percentage=50, method='percentage')
IDOA_vector_shuffled = IDOA_object_shuffled.calc_idoa_vector()
IDOA_object_SparseDOSSA2 = IDOA(SparseDOSSA2_stool_cohort_part, SparseDOSSA2_stool_cohort_part,
                                min_overlap=0.9, max_overlap=1, zero_overlap=0,
                                identical=True, min_num_points=0, percentage=50, method='percentage')
IDOA_vector_SparseDOSSA2 = IDOA_object_SparseDOSSA2.calc_idoa_vector()

# Plot IDOA histograms for all groups

num_bins = 10

bin_size = (max(max(IDOA_vector_real), max(IDOA_vector_shuffled)) - min(min(IDOA_vector_real), min(
    IDOA_vector_shuffled))) / num_bins

histogram_trace_real = go.Histogram(
    x=IDOA_vector_real,
    xbins=dict(start=min(IDOA_vector_real), end=max(IDOA_vector_real), size=bin_size),
    opacity=0.5,
    marker=dict(color='blue'),
    name='Real',
    histnorm='probability density'
)

histogram_trace_MIDAS = go.Histogram(
    x=IDOA_vector_MIDAS,
    xbins=dict(start=min(IDOA_vector_MIDAS), end=max(IDOA_vector_MIDAS), size=bin_size),
    opacity=0.5,
    marker=dict(color='chartreuse'),
    name='MIDAS',
    histnorm='probability density'
)

histogram_trace_SparseDOSSA2 = go.Histogram(
    x=IDOA_vector_SparseDOSSA2,
    xbins=dict(start=min(IDOA_vector_SparseDOSSA2), end=max(IDOA_vector_SparseDOSSA2), size=bin_size),
    opacity=0.5,
    marker=dict(color='red'),
    name='SparseDOSSA2',
    histnorm='probability density'
)

histogram_trace_shuffled = go.Histogram(
    x=IDOA_vector_shuffled,
    xbins=dict(start=min(IDOA_vector_shuffled), end=max(IDOA_vector_shuffled), size=bin_size),
    opacity=0.5,
    marker=dict(color='grey'),
    name='shuffled',
    histnorm='probability density'
)

layout = go.Layout(
    xaxis={
        'title': {"text": 'IDOA', "font": {"size": 30, "family": "Computer Modern"}},
        'zeroline': False,
        'showgrid': False,
        "showline": True,
        "linewidth": 2,
        'tickfont': dict(size=20),
    },
    yaxis={
        'title': {"text": 'Density', 'font': {'size': 30, "family": "Computer Modern"}},
        'showgrid': False,
        "showline": True,
        "linewidth": 2,
        'tickfont': dict(size=20)
    },
    legend=dict(x=0, y=1, font=dict(size=25, family="latex")),
    width=700,
    height=700,
    barmode='overlay',
)

fig = go.Figure(data=[histogram_trace_real, histogram_trace_MIDAS, histogram_trace_SparseDOSSA2,
                      histogram_trace_shuffled], layout=layout)

fig.show()