from Functions import *
from GLV_model import Glv
from IDOA import IDOA
import plotly.graph_objs as go

# Variables
time_span = 200
max_step = 0.5
delta = 0.00001
num_samples = 100
num_species = 100
r = np.random.uniform(0, 1, num_species)
s = np.ones(num_species)
low_factor = 0.5

# Create initial conditions for four models, with interactions, without interactions,
# low interactions and reference model
Y_no_interactions = calc_initial_conditions(num_species, num_samples)
Y_interactions = calc_initial_conditions(num_species, num_samples)
Y_low_interactions = calc_initial_conditions(num_species, num_samples)
Y_ref = calc_initial_conditions(num_species, num_samples)

# Create three corresponding interaction matrices
matrix_no_interactions = np.zeros([num_species, num_species])
matrix_interactions = create_interaction_matrix(num_species, delta_int=0.05, p=0.25)
matrix_low_interactions = matrix_interactions * low_factor

# Create a GLV model for the four models
glv_no_interactions = Glv(n_samples=num_samples, n_species=num_species, delta=delta, r=r, s=s,
                          interaction_matrix=matrix_no_interactions, initial_cond=Y_no_interactions,
                          final_time=time_span, max_step=max_step)
glv_no_interactions_results = glv_no_interactions.solve()

glv_interactions = Glv(n_samples=num_samples, n_species=num_species, delta=delta, r=r, s=s,
                       interaction_matrix=matrix_interactions, initial_cond=Y_interactions,
                       final_time=time_span, max_step=max_step)
glv_interactions_results = glv_interactions.solve()

glv_low_interactions = Glv(n_samples=num_samples, n_species=num_species, delta=delta, r=r, s=s,
                           interaction_matrix=matrix_low_interactions, initial_cond=Y_low_interactions,
                           final_time=time_span, max_step=max_step)
glv_low_interactions_results = glv_low_interactions.solve()

glv_ref = Glv(n_samples=num_samples, n_species=num_species, delta=delta, r=r, s=s,
              interaction_matrix=matrix_interactions, initial_cond=Y_ref,
              final_time=time_span, max_step=max_step)
glv_ref_results = glv_ref.solve()

# Calculate the IDOA vector for the three models w.r.t the reference model
IDOA_object_no_interactions = IDOA(glv_ref_results, glv_no_interactions_results, min_num_points=10, min_overlap=0.65,
                                   max_overlap=1, method='min_max')
IDOA_vector_no_interactions = IDOA_object_no_interactions.calc_idoa_vector()

IDOA_object_interactions = IDOA(glv_ref_results, glv_interactions_results, min_num_points=10, min_overlap=0.65,
                                   max_overlap=1, method='min_max')
IDOA_vector_interactions = IDOA_object_interactions.calc_idoa_vector()

IDOA_object_low_interactions = IDOA(glv_ref_results, glv_low_interactions_results, min_num_points=10, min_overlap=0.65,
                                   max_overlap=1, method='min_max')
IDOA_vector_low_interactions = IDOA_object_low_interactions.calc_idoa_vector()

# Create Histogram traces for each dataset
trace1 = go.Histogram(x=IDOA_vector_no_interactions, name='No interactions (ε=0)', histnorm='probability density', nbinsx=20,
                      opacity=0.7)
trace2 = go.Histogram(x=IDOA_vector_low_interactions, name='Weak interactions (ε=0.5)', histnorm='probability density', nbinsx=20,
                      opacity=0.7)
trace3 = go.Histogram(x=IDOA_vector_interactions, name='Strong interactions (ε=1)', histnorm='probability density', nbinsx=20,
                      opacity=0.7)


layout = go.Layout(
    xaxis=dict(title='IDOA', tickfont=dict(family='latex', size=30)),
    yaxis=dict(title='Probability Density', tickfont=dict(family='latex', size=30), showticklabels=False),
    barmode='overlay',
    xaxis_title_font=dict(family='latex', size=40),
    yaxis_title_font=dict(family='latex', size=40),
    width=1000,
    height=1000,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    legend=dict(x=0, y=1, font=dict(family='latex', size=30)),
    paper_bgcolor='white',
    plot_bgcolor='white',
)

layout.xaxis.linecolor = 'black'
layout.yaxis.linecolor = 'black'

layout.xaxis.linewidth = 2
layout.yaxis.linewidth = 2

fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)

fig.show()

# Calculate the mean Bray Curtis vector for the three models w.r.t the reference model
BC_vector_no_interactions = calc_bray_curtis_dissimilarity(glv_ref_results, glv_no_interactions_results)

BC_vector_interactions = calc_bray_curtis_dissimilarity(glv_ref_results, glv_interactions_results)

BC_vector_low_interactions = calc_bray_curtis_dissimilarity(glv_ref_results, glv_low_interactions_results)

# Create Histogram traces for each dataset
trace1 = go.Histogram(x=BC_vector_no_interactions, name='No interactions (ε=0)', histnorm='probability density', nbinsx=20,
                      opacity=0.7)
trace2 = go.Histogram(x=BC_vector_low_interactions, name='Weak interactions (ε=0.5)', histnorm='probability density', nbinsx=20,
                      opacity=0.7)
trace3 = go.Histogram(x=BC_vector_interactions, name='Strong interactions (ε=1)', histnorm='probability density', nbinsx=20,
                      opacity=0.7)


layout = go.Layout(
    xaxis=dict(title='Mean Bray Curtis', tickfont=dict(family='latex', size=30)),
    yaxis=dict(title='Probability Density', tickfont=dict(family='latex', size=30), showticklabels=False),
    barmode='overlay',
    xaxis_title_font=dict(family='latex', size=40),
    yaxis_title_font=dict(family='latex', size=40),
    width=1000,
    height=1000,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    legend=dict(x=0, y=1, font=dict(family='latex', size=30)),
    paper_bgcolor='white',
    plot_bgcolor='white',
)

layout.xaxis.linecolor = 'black'
layout.yaxis.linecolor = 'black'

layout.xaxis.linewidth = 2
layout.yaxis.linewidth = 2

fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)

fig.show()