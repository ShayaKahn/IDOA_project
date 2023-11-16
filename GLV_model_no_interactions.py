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

Y = clac_set_of_initial_conditions(num_species, num_samples)

matrix = np.zeros([num_species, num_species])

glv = Glv(n_samples=num_samples, n_species=num_species, delta=delta, r=r, s=s,
          interaction_matrix=matrix, initial_cond=Y, final_time=time_span, max_step=max_step)

glv_results = glv.solve()

IDOA_object = IDOA(glv_results, glv_results, identical=True, percentage=50, min_num_points=0, min_overlap=0.5,
                   max_overlap=1, zero_overlap=0, method='min_max_zero')
IDOA_vector = IDOA_object.calc_idoa_vector()

# Create a histogram plot
histogram = go.Histogram(x=IDOA_vector, nbinsx=10, histnorm='probability', opacity=0.75)

# Create a layout with customized axis labels and font, and remove the grid
layout = go.Layout(
    xaxis=dict(
        title='$\\mathrm{IDOA}$',  # LaTeX font for the x-axis title
        tickfont=dict(family='Computer Modern', size=18),  # Increase tick font size
        titlefont=dict(family='Computer Modern', size=24),  # Increase title font size
        range=[-0.01, 0.01],
        showgrid=False
    ),
    yaxis=dict(
        title='$\\mathrm{Density}$',  # LaTeX font for the y-axis title
        tickfont=dict(family='Computer Modern', size=18),  # Increase tick font size
        titlefont=dict(family='Computer Modern', size=24),  # Increase title font size
        showgrid=False
    ),
    width=600,
    height=600
)

# Create a figure
fig = go.Figure(data=[histogram], layout=layout)

# Show the plot
fig.show()