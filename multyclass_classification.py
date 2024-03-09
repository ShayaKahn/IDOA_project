from IDOA import IDOA
import numpy as np
from GLV_model import Glv
import plotly.graph_objects as go
from Functions import *

# Variables
s = np.ones(100)
r = np.random.uniform(0, 1, 100)
time_span = 200
max_step = 0.5
delta = 1e-3
num_samples = 100
num_species = 100

# Calculate interaction matrix
A = create_interaction_matrix(num_species)

# Set initial conditions
Y0 = calc_initial_conditions(num_species, num_samples)
Y1 = calc_initial_conditions(num_species, num_samples)
Y2 = calc_initial_conditions(num_species, num_samples)

interaction_matrix_0 = create_interaction_matrix(num_species)
interaction_matrix_1 = create_interaction_matrix(num_species)
interaction_matrix_2 = create_interaction_matrix(num_species)

# Initiate Glv objects

glv_0 = Glv(n_samples=num_samples, n_species=num_species, delta=delta, r=r, s=s,
            interaction_matrix=interaction_matrix_0, initial_cond=Y0, final_time=time_span, max_step=max_step)
glv_1 = Glv(n_samples=num_samples, n_species=num_species, delta=delta, r=r, s=s,
            interaction_matrix=interaction_matrix_1, initial_cond=Y1, final_time=time_span, max_step=max_step)
glv_2 = Glv(n_samples=num_samples, n_species=num_species, delta=delta, r=r, s=s,
            interaction_matrix=interaction_matrix_2, initial_cond=Y2, final_time=time_span, max_step=max_step)

# Solve the models

glv_0_results = glv_0.solve()
glv_1_results = glv_1.solve()
glv_2_results = glv_2.solve()

# Set IDOA object for each model where the reference is changes.

IDOA_object_0_0 = IDOA(glv_0_results, glv_0_results, percentage=50, min_num_points=0, min_overlap=0.5, max_overlap=1,
                       method='min_max')
IDOA_object_1_0 = IDOA(glv_1_results, glv_0_results, percentage=50, min_num_points=0, min_overlap=0.5, max_overlap=1,
                       method='min_max')
IDOA_object_2_0 = IDOA(glv_2_results, glv_0_results, percentage=50, min_num_points=0, min_overlap=0.5, max_overlap=1,
                       method='min_max')
IDOA_object_0_1 = IDOA(glv_0_results, glv_1_results, percentage=50, min_num_points=0, min_overlap=0.5, max_overlap=1,
                       method='min_max')
IDOA_object_1_1 = IDOA(glv_1_results, glv_1_results, percentage=50, min_num_points=0, min_overlap=0.5, max_overlap=1,
                       method='min_max')
IDOA_object_2_1 = IDOA(glv_2_results, glv_1_results, percentage=50, min_num_points=0, min_overlap=0.5, max_overlap=1,
                       method='min_max')
IDOA_object_0_2 = IDOA(glv_0_results, glv_2_results, percentage=50, min_num_points=0, min_overlap=0.5, max_overlap=1,
                       method='min_max')
IDOA_object_1_2 = IDOA(glv_1_results, glv_2_results, percentage=50, min_num_points=0, min_overlap=0.5, max_overlap=1,
                       method='min_max')
IDOA_object_2_2 = IDOA(glv_2_results, glv_2_results, percentage=50, min_num_points=0, min_overlap=0.5, max_overlap=1,
                       method='min_max')

# Calculate IDOA values for each IDOA object.

IDOA_vector_0_0 = IDOA_object_0_0.calc_idoa_vector()
IDOA_vector_1_0 = IDOA_object_1_0.calc_idoa_vector()
IDOA_vector_2_0 = IDOA_object_2_0.calc_idoa_vector()
IDOA_vector_0_1 = IDOA_object_0_1.calc_idoa_vector()
IDOA_vector_1_1 = IDOA_object_1_1.calc_idoa_vector()
IDOA_vector_2_1 = IDOA_object_2_1.calc_idoa_vector()
IDOA_vector_0_2 = IDOA_object_0_2.calc_idoa_vector()
IDOA_vector_1_2 = IDOA_object_1_2.calc_idoa_vector()
IDOA_vector_2_2 = IDOA_object_2_2.calc_idoa_vector()

# Create 3D matrix for each Glv model.

class_mat_0 = np.vstack([IDOA_vector_0_0, IDOA_vector_1_0, IDOA_vector_2_0])
class_mat_1 = np.vstack([IDOA_vector_0_1, IDOA_vector_1_1, IDOA_vector_2_1])
class_mat_2 = np.vstack([IDOA_vector_0_2, IDOA_vector_1_2, IDOA_vector_2_2])

# Plot the 3D matrices
def plot_3d_matrices(matrices, titles=('IDOA w.r.t Dynamics A', 'IDOA w.r.t Dynamics B', 'IDOA w.r.t Dynamics C')):
    # Create a Plotly figure
    fig = go.Figure()

    # Define colors for each matrix
    colors = ['red', 'blue', 'grey']

    names = ['Dynamics A', 'Dynamics B', 'Dynamics C']

    for i, matrix in enumerate(matrices):
        # Extract the x, y, and z coordinates from the matrix
        x = matrix[0]
        y = matrix[1]
        z = matrix[2]

        # Create a scatter plot for each matrix with the specified color
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=4,
                color=colors[i],
                opacity=0.8,
            ),
            name=f'{names[i]}'
        ))

    axis_font_size = 20
    fig.update_layout(
        scene=dict(
            xaxis_title=titles[0],
            yaxis_title=titles[1],
            zaxis_title=titles[2],
            xaxis=dict(
                zeroline=False,
                showbackground=True,
                backgroundcolor="white",
                showline=True,
                linecolor="black",
                gridwidth=1,
                gridcolor="black",
                title_font=dict(size=axis_font_size),  # Increase the font size for the x-axis title
                tickfont=dict(size=15)  # Increase the font size for tick labels
            ),
            yaxis=dict(
                zeroline=False,
                showbackground=True,
                backgroundcolor="white",
                showline=True,
                linecolor="black",
                gridwidth=1,
                gridcolor="black",
                title_font=dict(size=axis_font_size),  # Increase the font size for the y-axis title
                tickfont=dict(size=15)  # Increase the font size for tick labels
            ),
            zaxis=dict(
                zeroline=False,
                showbackground=True,
                backgroundcolor="white",
                showline=True,
                linecolor="black",
                gridwidth=1,
                gridcolor="black",
                title_font=dict(size=axis_font_size),  # Increase the font size for the z-axis title
                tickfont=dict(size=15)  # Increase the font size for tick labels
            ),
            aspectmode="cube",
            aspectratio=dict(x=1, y=1, z=1)
        )
    )

    # Set the location of the legend to upper left and increase legend font size
    fig.update_layout(
        legend=dict(
            x=0.2,
            y=0.7,
            xanchor='left',
            yanchor='top',
            title_font=dict(size=axis_font_size),  # Increase the font size for the legend
        )
    )

    # Set the font for the axis and legend to Computer Modern
    fig.update_layout(
        font=dict(
            family='Computer Modern',
        )
    )

    # Set the size of the plot
    fig.update_layout(
        width=1000,
        height=1000,
        autosize=False
    )

    # Show the plot
    fig.show()

plot_3d_matrices([class_mat_0, class_mat_1, class_mat_2])

# Bray-Curtis distance
def mean_bray_curtis_distances(cohort_first, cohort_second):

    distances = []

    for row_second in cohort_second:
        row_distances = []

        for row_first in cohort_first:
            if not np.array_equal(row_first, row_second):
                distance = braycurtis(row_first, row_second)#braycurtis(row_first, row_second)
                row_distances.append(distance)

        if row_distances:
            mean_distance = np.mean(row_distances)
            distances.append(mean_distance)

    return distances

# Calculate mean Bray Curtis distances.

BC_vector_0_0 = mean_bray_curtis_distances(glv_0_results, glv_0_results)
BC_vector_1_0 = mean_bray_curtis_distances(glv_1_results, glv_0_results)
BC_vector_2_0 = mean_bray_curtis_distances(glv_2_results, glv_0_results)
BC_vector_0_1 = mean_bray_curtis_distances(glv_0_results, glv_1_results)
BC_vector_1_1 = mean_bray_curtis_distances(glv_1_results, glv_1_results)
BC_vector_2_1 = mean_bray_curtis_distances(glv_2_results, glv_1_results)
BC_vector_0_2 = mean_bray_curtis_distances(glv_0_results, glv_2_results)
BC_vector_1_2 = mean_bray_curtis_distances(glv_1_results, glv_2_results)
BC_vector_2_2 = mean_bray_curtis_distances(glv_2_results, glv_2_results)

# Create 3D matrix for each Glv model.

class_mat_0_bc = np.vstack([BC_vector_0_0, BC_vector_1_0, BC_vector_2_0])
class_mat_1_bc = np.vstack([BC_vector_0_1, BC_vector_1_1, BC_vector_2_1])
class_mat_2_bc = np.vstack([BC_vector_0_2, BC_vector_1_2, BC_vector_2_2])

# Plot the 3D matrices.

plot_3d_matrices([class_mat_0_bc, class_mat_1_bc, class_mat_2_bc], titles=('Mean Bray Curtis w.r.t Dynamics A',
                                                                           'Mean Bray Curtis w.r.t Dynamics B',
                                                                           'Mean Bray Curtis w.r.t Dynamics C'))
