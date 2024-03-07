import pandas as pd
import numpy as np
import os
os.chdir(r'C:\Users\shaya\OneDrive\Desktop\IDOA\Diet_effects_on_IDOA')
diet_data = pd.read_excel('Diet_data_filtered.xlsx', header=None)
diet_data = diet_data.values
diet_otu = pd.read_excel('Diet_otu.xlsx', header=None)
diet_otu = diet_otu.values.T

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

diet_otu = normalize_cohort(diet_otu)


# Create two subsets of diet_otu for each row in diet_otu. The first subset represents the half of the samples in
# diet_otu that have the most similar diet pattern (determined using the L2 norm), and the other half is the second
# subset.

from sklearn.decomposition import PCA

def get_all_similar_and_dissimilar_subsets(X, X_otu, PC=False, components=2):
    """
    Given a datasets X and X_otu, this function returns n sets of subsets for each row in X_otu:
    1) A subset containing vectors in X_otu that their corresponding vectors in X are more similar to the 
       test vector in X.
    2) A subset containing vectors in X_otu that their corresponding vectors in X are least similar to the 
       test vector in X.
    
    Parameters:
    - X: numpy array of shape (n_samples, n_features), represent the diet matrix.
    - X_otu: numpy array of shape (n_samples, n_features), represent the otu matrix.
    
    Returns:
    - all_subsets: List of tuples, each containing two numpy arrays (similar_subset, dissimilar_subset),
      the subsets are taken from the otu matrix.
    """
    all_subsets = []
    similar_indexes_container = []
    
    if PC:
        pca = PCA(n_components=components)
        X = pca.fit_transform(X)
    
    for v in X:
            
        # Compute Euclidean distances between v and all vectors in X
        distances = np.linalg.norm(X - v, axis=1)
        
        # Get sorted indices based on distances
        sorted_indices = np.argsort(distances)
        
        n_samples = X.shape[0]
        
        # Determine the size of each subset
        half_n_samples = n_samples // 2
        
        # Get indices for similar and dissimilar subsets, we take the similar_indices from 1 to avoid
        # calculations of similarity between each sample with itself.
        similar_indices = sorted_indices[1:half_n_samples]
        dissimilar_indices = sorted_indices[-half_n_samples:]
        
        similar_indexes_container.append(similar_indices)
        
        # Create similar and dissimilar subsets
        similar_subset = X_otu[similar_indices]
        dissimilar_subset = X_otu[dissimilar_indices]
        
        all_subsets.append((similar_subset, dissimilar_subset))
    
    return all_subsets, similar_indexes_container

subsets_list, similar_indices = get_all_similar_and_dissimilar_subsets(diet_data, diet_otu, PC=True)

import os
os.chdir(r'C:\Users\shaya\OneDrive\Desktop\IDOA_project')
from IDOA import IDOA

# Calculation of the IDOA values using the top 50% overlap value points on the dissimilarity-overlap space. 

IDOA_similar_diet = []
IDOA_dissimilar_diet = []
D_O_container_similar = []
D_O_container_dissimilar = []
for sample, subsets in zip(diet_otu, subsets_list):
    IDOA_object_similar_diet = IDOA(subsets[0], sample, min_overlap=0.9, max_overlap=1, min_num_points=0,
                                    percentage=40, method='percentage')
    IDOA_similar_diet.append(IDOA_object_similar_diet.calc_idoa_vector())
    D_O_container_similar.append(
        IDOA_object_similar_diet.d_o_list_unconst)
    IDOA_object_dissimilar_diet = IDOA(subsets[1], sample, min_overlap=0.9, max_overlap=1, min_num_points=0,
                                    percentage=40, method='percentage')
    IDOA_dissimilar_diet.append(IDOA_object_dissimilar_diet.calc_idoa_vector())
    D_O_container_dissimilar.append(
        IDOA_object_dissimilar_diet.d_o_list_unconst)
    
# Delete Outlier
del IDOA_similar_diet[-6]
del IDOA_dissimilar_diet[-6]

import plotly.graph_objs as go

# Use arbitrary example
x_1 = D_O_container_similar[18][0][0]
y_1 = D_O_container_similar[18][0][1]
x_2 = D_O_container_dissimilar[18][0][0]
y_2 = D_O_container_dissimilar[18][0][1]

fig = go.Figure()

fig.add_trace(go.Scatter(x=x_1, y=y_1, 
                         mode='markers', marker=dict(color='red')))

fig.add_trace(go.Scatter(x=x_2, y=y_2, 
                         mode='markers', marker=dict(color='blue')))

# Calculate the IDOA for the top 50% of the points for the similar sub-cohort
top_x1_indices = np.where(x_1 >= np.percentile(x_1, 50))
x1_top = np.array(x_1)[top_x1_indices]
y1_top = np.array(y_1)[top_x1_indices]
slope1, intercept1 = np.polyfit(x1_top, y1_top, 1)
y1_fit = slope1 * x1_top + intercept1
fig.add_trace(go.Scatter(x=x1_top, y=y1_fit, mode='lines', line=dict(color='red', width=4)))

# Calculate the IDOA for the top 50% of the points for the dissimilar sub-cohort
top_x2_indices = np.where(x_2 >= np.percentile(x_2, 50))
x2_top = np.array(x_2)[top_x2_indices]
y2_top = np.array(y_2)[top_x2_indices]
slope2, intercept2 = np.polyfit(x2_top, y2_top, 1)
y2_fit = slope2 * x2_top + intercept2
fig.add_trace(go.Scatter(x=x2_top, y=y2_fit, mode='lines', line=dict(color='blue', width=4)))

# Layout
fig.update_layout(
    width=600,
    height=600,
    xaxis_title="Overlap",
    yaxis_title="Dissimilarity",
    xaxis=dict(
        showgrid=False,  
        zeroline=False,
        showline=True,
        linecolor='black',
        range=[0.15, 0.75]  
    ),
    yaxis=dict(
        showgrid=False, 
        zeroline=False,
        showline=True,
        linecolor='black',
        range=[0.15, 0.75]   
    ),
    font=dict(
        size=18
    ),
    showlegend=False,
    plot_bgcolor='white'
)
fig.update_xaxes(title_font=dict(size=24, family="Computer Modern"))
fig.update_yaxes(title_font=dict(size=24, family="Computer Modern"))

fig.show()

# Correlation scatterplot

scatter_trace = go.Scatter(
    x=IDOA_similar_diet,
    y=IDOA_dissimilar_diet,
    mode='markers',
)

# Create layout
layout = go.Layout(
    width=600,
    height=600,
    xaxis=dict(
        title="IDOA w.r.t similar diet",
        showgrid=False,
        zeroline=True,
        showline=True,
        linecolor='black',
        zerolinecolor="black",
        range=[-0.9, 0.9],
        titlefont=dict(size=24, family="Computer Modern"),
    ),
    yaxis=dict(
        title="IDOA w.r.t dissimilar diet",
        showgrid=False,
        zeroline=True,
        zerolinecolor="black",
        showline=True,
        linecolor='black',
        range=[-0.9, 0.9],
        titlefont=dict(size=24, family="Computer Modern"),
    ),
    font=dict(
        family="Arial, monospace",
        size=18,
    ),
    plot_bgcolor='white'
)

# Create figure
fig = go.Figure(data=[scatter_trace], layout=layout)

# Add dashed black equality line
fig.add_shape(
    type="line",
    x0=-0.9,
    x1=0.9,
    y0=-0.9,
    y1=0.9,
    line=dict(color="black", width=2, dash="dash"),
)

# Show the plot
fig.show()

# PCoA Plot

similar_ind = similar_indices[18]

pca = PCA(n_components=2)
X = pca.fit_transform(diet_data)

df = pd.DataFrame(data=X, columns=['PC1', 'PC2'])

df['color'] = 'blue'

df.loc[similar_ind, 'color'] = 'red'

fig = go.Figure()

fig.add_trace(go.Scatter(x=df['PC1'], y=df['PC2'], mode='markers',
                         marker=dict(color=df['color']),
                         showlegend=False))

fig.add_trace(go.Scatter(x=[df.loc[18, 'PC1']], y=[df.loc[18, 'PC2']], mode='markers',
                         marker=dict(color='black', symbol='circle', size=16),
                         showlegend=False))

fig.update_layout(
    width=600,
    height=600,
    xaxis_title="PC1", 
    yaxis_title="PC2",  
    xaxis=dict(
        showgrid=False,  
        zeroline=False,
        showline=True,
        linecolor='black'
    ),
    yaxis=dict(
        showgrid=False,  
        zeroline=False,
        showline=True,
        linecolor='black'
    ),
    font=dict(
        family="Arial, monospace", 
        size=18
    ),
    plot_bgcolor='white'
)

fig.update_xaxes(title_font=dict(size=24, family="Computer Modern"))
fig.update_yaxes(title_font=dict(size=24, family="Computer Modern"))

fig.show()