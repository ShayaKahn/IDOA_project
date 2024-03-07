import os
from dash import Dash, html, dcc
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import MDS
from scipy.spatial.distance import cdist
from Functions import *
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from IDOA import IDOA
from sklearn.metrics import roc_auc_score

def normalize_data(data):
    """
    :param data: Matrix of data
    :return: Normalized matrix of the data.
    """
    norm_factors = np.sum(data, axis=0)
    norm_data = np.array([data[:, i] / norm_factors[i] for i in range(0, np.size(norm_factors))])
    return norm_data.T

########## Load data - ASD ##########

# Load already calculated distance values
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

# Load DOC data and confusion matrices that already calculated (cutoff = 0.5 for IDOA)
df_DOC_control_asd = pd.read_csv('Doc_mat_control.csv', header=None)
DOC_control_asd = df_DOC_control_asd.to_numpy()
df_DOC_asd = pd.read_csv('Doc_mat_ASD.csv', header=None)
DOC_asd = df_DOC_asd.to_numpy()

# Load ASD data
ASD_data = pd.read_csv('ASD_data.csv', header=None)
ASD_data = ASD_data.to_numpy()
control_data = pd.read_csv('control_data.csv', header=None)
control_data = control_data.to_numpy()

# Normalize ASD data
ASD_data_norm = normalize_cohort(ASD_data.T)
control_data_norm = normalize_cohort(control_data.T)

idoa_obj = IDOA(control_data_norm, ASD_data_norm[100, :], method='min_max')
IDOA_val = idoa_obj.calc_idoa_vector()
info = idoa_obj.additional_info()
d_o_list = info['d_o_list'][0]
new_overlap_vector = d_o_list[0, :]
new_dissimilarity_vector = d_o_list[1, :]

sm_x_control_asd, sm_y_control_asd = sm_lowess(endog=DOC_control_asd[0][:], exog=DOC_control_asd[1][:],
                                               frac=0.3, it=3, return_sorted=True).T
sm_x_asd_idoa, sm_y_asd_idoa = sm_lowess(endog=new_dissimilarity_vector, exog=new_overlap_vector, frac=0.3, it=3,
                                         return_sorted=True).T
sm_x_control, sm_y_control = sm_lowess(endog=DOC_control_asd[0][:], exog=DOC_control_asd[1][:], frac=0.9, it=3,
                                       return_sorted=True).T
sm_x_asd_idoa_control, sm_y_asd_idoa_control = sm_lowess(endog=new_dissimilarity_vector, exog=new_overlap_vector,
                                                         frac=0.9, it=3, return_sorted=True).T
sm_x_asd, sm_y_asd = sm_lowess(endog=DOC_asd[0][:], exog=DOC_asd[1][:], frac=0.9, it=3, return_sorted=True).T

# plot the graph of moving averages
cutt_control = find_cutoff(sm_x_control, sm_y_control, 2)
cutt_asd = find_cutoff(sm_x_asd, sm_y_asd, 2)

# Define parameters for IDOA
per = 5
min_num_points = 10
min_overlap_control = 0.5
min_overlap_ASD = 0.5
maximal = 1

# Create IDOA vectors for the analysis
idoa_control_asd_vector_object = IDOA(control_data_norm, ASD_data_norm, percentage=per, min_num_points=min_num_points,
                                      min_overlap=min_overlap_control, max_overlap=maximal, method='min_max')
idoa_control_asd_vector = idoa_control_asd_vector_object.calc_idoa_vector()
over_cutoff_control_asd = idoa_control_asd_vector_object.over_cutoff
idoa_asd_control_vector_object = IDOA(ASD_data_norm, control_data_norm, percentage=per, min_num_points=min_num_points,
                                      min_overlap=min_overlap_ASD, max_overlap=maximal, method='min_max')
idoa_asd_control_vector = idoa_asd_control_vector_object.calc_idoa_vector()
idoa_asd_asd_vector_object = IDOA(ASD_data_norm, ASD_data_norm, percentage=per, min_num_points=min_num_points,
                                  min_overlap=min_overlap_ASD, max_overlap=maximal, method='min_max')
idoa_asd_asd_vector = idoa_asd_asd_vector_object.calc_idoa_vector()
idoa_control_control_vector_asd_object = IDOA(control_data_norm, control_data_norm, percentage=per,
                                              min_num_points=min_num_points, min_overlap=min_overlap_control,
                                              max_overlap=maximal, method='min_max')
idoa_control_control_vector_asd = idoa_control_control_vector_asd_object.calc_idoa_vector()


# Calculate AUC values
dist_control = np.concatenate((dist_control_control_vector_asd, dist_control_asd_vector))
labels_control_dist = np.concatenate((np.zeros(dist_control_control_vector_asd.size),
                                      np.ones(dist_control_asd_vector.size)))
auc_control_dist = roc_auc_score(labels_control_dist, dist_control)
dist_asd = np.concatenate((dist_asd_asd_vector, dist_asd_control_vector))
labels_asd_dist = np.concatenate((np.zeros(dist_asd_asd_vector.size),
                                  np.ones(dist_asd_control_vector.size)))
auc_asd_dist = roc_auc_score(labels_asd_dist, dist_asd)
idoa_control = np.concatenate((idoa_control_control_vector_asd, idoa_control_asd_vector))
labels_control_idoa = np.concatenate((np.zeros(idoa_control_control_vector_asd.size),
                                      np.ones(idoa_control_asd_vector.size)))
auc_control_idoa = roc_auc_score(labels_control_idoa, idoa_control)
idoa_asd = np.concatenate((idoa_asd_asd_vector, idoa_asd_control_vector))
labels_asd_idoa = np.concatenate((np.zeros(idoa_asd_asd_vector.size),
                                  np.ones(idoa_asd_control_vector.size)))
auc_asd_idoa = roc_auc_score(labels_asd_idoa, idoa_asd)

# Plots

app = Dash(__name__)

scatter_idoa_samp_ASD = go.Scattergl(x=new_overlap_vector, y=new_dissimilarity_vector, marker={
                                      "color": "blue", "size": 5}, showlegend=False, mode="markers")
scatter_DOC_control_asd = go.Scatter(x=DOC_control_asd[1][:], y=DOC_control_asd[0][:], marker={
                                     "color": "red", "size": 2}, showlegend=False, mode="markers")
fig_control_asd = [scatter_DOC_control_asd, go.Scattergl(x=sm_x_control_asd, y=sm_y_control_asd,
                   line={"color": "red", "dash": 'solid', "width": 4.5}, showlegend=False),
                   go.Scattergl(x=sm_x_asd_idoa, y=sm_y_asd_idoa, line={"color": "blue", "dash": 'solid',
                   "width": 4.5}, showlegend=False), scatter_idoa_samp_ASD]
vertical_line_control = go.Scatter(x=[cutt_control, cutt_control], y=[0, 1], mode='lines', line=dict(color='black',
                                   width=4), showlegend=False)
percentage_control = np.sum((cutt_control <= DOC_control_asd[1][:]))/np.size(DOC_control_asd[1][:])
percentage_asd = np.sum((cutt_asd <= DOC_asd[1][:]))/np.size(DOC_asd[1][:])
hist_values, bin_edges = np.histogram(DOC_control_asd[1][:], bins=100)
hist_values = hist_values/(np.sum(hist_values))
histogram_control = go.Scatter(x=bin_edges, y=hist_values, mode='lines', fill='tozeroy', line=dict(color='red',
                               width=1), showlegend=False)
fig_control = [scatter_DOC_control_asd, histogram_control, vertical_line_control, go.Scattergl(x=sm_x_control,
               y=sm_y_control, line={"color": "darkred", "dash": 'solid', "width": 4.5}, showlegend=False)]
scatter_DOC_asd = go.Scatter(x=DOC_asd[1][:], y=DOC_asd[0][:], marker={"color": "blue", "size": 2}, showlegend=False,
                             mode="markers")
vertical_line_asd = go.Scatter(x=[cutt_asd, cutt_asd], y=[0, 1], mode='lines', line=dict(color='black', width=4),
                               showlegend=False)
hist_values, bin_edges = np.histogram(DOC_asd[1][:], bins=100)
hist_values = hist_values/(np.sum(hist_values))
histogram_ASD = go.Scatter(x=bin_edges, y=hist_values, mode='lines', fill='tozeroy', line=dict(color='blue', width=1),
                           showlegend=False)
fig_ASD = [scatter_DOC_asd, histogram_ASD, vertical_line_asd, go.Scattergl(x=sm_x_asd, y=sm_y_asd, line={"color":
           "darkblue", "dash": 'solid', "width": 4.5}, showlegend=False)]
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
hist_dist_control_asd_vector, bins_dist_control_asd_vector = np.histogram(dist_control_asd_vector, bins=8, density=True)
hist_dist_control_control_vector, bins_dist_control_control_vector = np.histogram(dist_control_control_vector_asd,
                                                                                  bins=8, density=True)
hist_idoa_control_asd_vector, bins_idoa_control_asd_vector = np.histogram(idoa_control_asd_vector, bins=8, density=True)
hist_idoa_control_control_vector, bins_idoa_control_control_vector = np.histogram(idoa_control_control_vector_asd,
                                                                                  bins=8, density=True)
hist_dist_control_asd = go.Histogram(x=dist_control_asd_vector, nbinsx=15, name='ASD', histnorm='probability density',
                                      marker=dict(color='blue', opacity=0.6))
hist_dist_control_control = go.Histogram(x=dist_control_control_vector_asd, nbinsx=15, name='Control',
                                         histnorm='probability density', marker=dict(color='red', opacity=0.8))
hist_idoa_control_asd = go.Histogram(x=idoa_control_asd_vector, nbinsx=15, name='ASD', histnorm='probability density',
                                      marker=dict(color='blue', opacity=0.6))
hist_idoa_control_control = go.Histogram(x=idoa_control_control_vector_asd, nbinsx=15, name='Control',
                                         histnorm='probability density', marker=dict(color='red', opacity=0.8))

hist_dist_asd_control_vector, bins_dist_asd_control_vector = np.histogram(dist_asd_control_vector, bins=10,
                                                                          density=True)
hist_dist_asd_asd_vector, bins_dist_asd_asd_vector = np.histogram(dist_asd_asd_vector,
                                                                  bins=8, density=True)
hist_idoa_asd_control_vector, bins_idoa_asd_control_vector = np.histogram(idoa_asd_control_vector, bins=10,
                                                                          density=True)
hist_idoa_asd_asd_vector, bins_idoa_asd_asd_vector = np.histogram(idoa_asd_asd_vector,
                                                                  bins=8, density=True)
hist_dist_asd_control = go.Histogram(x=dist_asd_control_vector, nbinsx=15, name='Control',
                                     histnorm='probability density', marker=dict(color='red', opacity=0.8))
hist_dist_asd_asd = go.Histogram(x=dist_asd_asd_vector, nbinsx=15, name='ASD',
                                     histnorm='probability density', marker=dict(color='blue', opacity=0.6))
hist_idoa_asd_control = go.Histogram(x=idoa_asd_control_vector, nbinsx=15, name='Control',
                                     histnorm='probability density', marker=dict(color='red', opacity=0.8))
hist_idoa_asd_asd = go.Histogram(x=idoa_asd_asd_vector, nbinsx=15, name='ASD',
                                     histnorm='probability density', marker=dict(color='blue', opacity=0.6))
scatter_IDOA_asd = go.Scatter(x=idoa_control_asd_vector, y=idoa_asd_asd_vector, marker={"color": "blue"}, name='ASD',
                              mode="markers", opacity=0.6)
scatter_IDOA_control_asd = go.Scatter(x=idoa_control_control_vector_asd, y=idoa_asd_control_vector, marker={
    "color": "red"}, name='Control', mode="markers", opacity=0.8)
line_idoa_asd = go.Scattergl(x=[-4.5, 1], y=[-4.5, 1], line={"color": "black", "dash": 'dash'}, mode="lines",
                             showlegend=False)
scatter_dist_asd = go.Scatter(x=dist_control_asd_vector, y=dist_asd_asd_vector, marker={"color": "blue"}, name='ASD',
                              mode="markers", opacity=0.6)
scatter_dist_control_asd = go.Scatter(x=dist_control_control_vector_asd, y=dist_asd_control_vector, marker={
    "color": "red"}, name='Control', mode="markers", opacity=0.8)
line_dist_asd = go.Scattergl(x=[0.35, 0.7], y=[0.35, 0.7], line={"color": "black", "dash": 'dash'},
                             mode="lines", showlegend=False)
scatter_asd = go.Scatter(x=dist_control_asd_vector, y=idoa_control_asd_vector,
                         marker={"color": "blue"}, name='ASD', mode="markers", opacity=0.6)
scatter_control_asd = go.Scatter(x=dist_control_control_vector_asd, y=idoa_control_control_vector_asd,
                                 marker={"color": "red"}, name='Control', mode="markers", opacity=0.8)
# Define the layout of the app
app.layout = html.Div([
    html.Div([
        html.H1(children='IDOA and Distances methods - ASD')], className='header'),
        html.Div([dcc.Graph(id='Distance Histograms Control', figure={'data': [hist_dist_control_control,
                 hist_dist_control_asd], 'layout': go.Layout(xaxis={'title': {'text':
                 'Mean distance to control', 'font': {'size': 25, 'family': 'latex'}}, 'zeroline': False,
                 'showgrid': False, 'showline': True, 'linewidth': 2,
                 'range': [0.38, 0.69]}, yaxis={'title': {'text': 'Density', 'font': {'size': 25,
                 'family': 'latex'}}, 'showgrid': False, 'showline': True, 'linewidth': 2, 'showticklabels': False,
                }, xaxis_tickfont=dict(size=20, family='latex'), yaxis_tickfont=dict(size=20, family='latex'),
                legend=dict(x=0.7, y=1, font=dict(size=17.5, family='latex')), barmode='overlay', width=500,
                                                             height=500)})], style={'width': '45%', 'display':
            'inline-block'}),
        html.Div([
                dcc.Graph(
                    id='Distance Histograms ASD',
                    figure={'data': [hist_dist_asd_control, hist_dist_asd_asd], 'layout': go.Layout(xaxis={'title': {
                            "text": 'Mean distance to ASD', "font": {"size": 25, 'family': 'latex'}}, 'zeroline': False,
                            'showgrid': False, "showline": True, "linewidth": 2, 'range': [0.37, 0.65]}, yaxis={
                            'title': {"text": 'Density', 'font': {'size': 25, 'family': 'latex'}}, 'showgrid': False,
                            "showline": True, "linewidth": 2, "showticklabels": False}, xaxis_tickfont=dict(size=20,
                            family='latex'), yaxis_tickfont=dict(size=20, family='latex'), legend=dict(x=0.7, y=1,
                            font=dict(size=17.5, family='latex')), barmode='overlay', width=500, height=500)})],
                            style={'width': '45%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='IDOA Histograms Control',
                    figure={'data': [hist_idoa_control_control, hist_idoa_control_asd], 'layout': go.Layout(xaxis={
                            'title': {"text": 'IDOA w.r.t control', "font": {"size": 25, 'family': 'latex'}},
                            'zeroline': False, 'showgrid': False, "showline": True, "linewidth": 2,
                            'range': [-3.1, 1.1]}, yaxis={'title': {"text": 'Density', 'font': {'size': 25, 'family':
                            'latex'}}, 'showgrid': False, "showline": True, "linewidth": 2, "showticklabels": False},
                            xaxis_tickfont=dict(size=20, family='latex'), yaxis_tickfont=dict(size=20, family='latex'),
                            barmode='overlay', legend=dict(x=0, y=1, font=dict(size=17.5, family='latex')), width=500,
                            height=500)})], style={'width': '40%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='IDOA Histograms ASD',
                    figure={'data': [hist_idoa_asd_control, hist_idoa_asd_asd], 'layout': go.Layout(xaxis={'range':
                            [-3.6, 1.1], 'title': {"text": 'IDOA w.r.t ASD', "font": {"size": 25, 'family': 'latex'}},
                            'zeroline': False, 'showgrid': False, "showline": True, "linewidth": 2}, yaxis={'title':
                            {"text": 'Density', 'font': {'size': 25, 'family': 'latex'}}, 'showgrid': False, "showline":
                            True, "linewidth": 2, "showticklabels": False}, xaxis_tickfont=dict(size=20,
                            family='latex'), barmode='overlay', yaxis_tickfont=dict(size=20, family='latex'),
                            legend=dict(x=0, y=1, font=dict(size=17.5, family='latex')), width=500, height=500)})],
                            style={'width': '45%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='IDOA - ASD',
                    figure={'data': [scatter_IDOA_control_asd, scatter_IDOA_asd, line_idoa_asd], 'layout': {'xaxis': {
                            'title': {"text": 'IDOA w.r.t control', "font": {"size": 25, 'family': 'latex'}},
                            "tickfont": {"size": 20, 'family': 'latex'}, 'zeroline': False, 'scaleratio': 1,
                            'scaleanchor': 'y', 'showgrid': False, "showline": True, "linewidth": 2}, 'yaxis': {'title':
                            {"text": 'IDOA w.r.t ASD', "font": {"size": 25, 'family': 'latex'}}, 'zeroline': False,
                            "tickfont": {"size": 20, 'family': 'latex'}, 'scaleratio': 1, 'scaleanchor': 'x',
                            'showgrid': False, "showline": True, "linewidth": 2}, 'legend': {'x': 0, 'y': 1, "font": {
                            "size": 17.5, 'family': 'latex'}}, "width": 500, "height": 500}})], style={'width': '48%',
                            'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='Combination IDOA - distances',
                    figure={'data': [scatter_control_asd, scatter_asd], 'layout': go.Layout(
                            xaxis={"title": {"text": 'Mean distance to control', "font": {"size": 25, 'family':
                            'latex'}}, 'showgrid' : False, 'zeroline': False, "showline": True,
                            "linewidth": 2, "range": [0.35, 0.7]}, yaxis={"title": {"text": 'IDOA w.r.t control',
                            "font": {"size": 25, 'family': 'latex'}}, 'showgrid': False, 'zeroline': False, "showline":
                            True, "linewidth": 2, "range": [-3, 1]}, xaxis_tickfont=dict(size=20, family='latex'),
                            yaxis_tickfont=dict(size=20, family='latex'), width=500, height=500, legend=dict(x=0.7, y=0,
                            font=dict(size=17.5, family='latex')))})], style={'display': 'flex', 'flexDirection':
                            'row'}),
            html.Div([
                dcc.Graph(
                    id='Distances - ASD',
                    figure={'data': [scatter_dist_control_asd, scatter_dist_asd, line_dist_asd], 'layout': {'xaxis':
                            {'title': {"text": 'Mean distance to control', "font": {"size": 25, 'family': 'latex'}},
                            'zeroline': False, 'scaleratio': 1, 'scaleanchor': 'y', 'showgrid': False, "showline": True,
                            "linewidth": 2, "tickfont": {"size": 20, 'family': 'latex'}},
                            'yaxis': {'title': {"text": 'Mean distance to ASD', "font": {"size": 25,
                            'family': 'latex'}}, 'zeroline': False, "showline": True, "linewidth": 2, 'scaleratio': 1,
                            'scaleanchor': 'x', 'showgrid': False, "tickfont": {"size": 20, 'family': 'latex'}},
                            'legend': {'x': 0, 'y': 1, "font": {"size": 17.5, 'family': 'latex'}},
                            'showgrid': False, "width": 500, "height": 500,}},)], style={'width': '48%', 'display':
                            'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='PCoA - ASD',
                    figure={'data': [PCoA_control_asd, PCoA_asd], 'layout': {'xaxis': {'title': {"text": 'PCo1',
                            "font": {"size": 25, 'family': 'latex'}}, 'zeroline': False, 'scaleratio': 1, 'scaleanchor':
                            'y', 'showgrid': False, "showline": True, "linewidth": 2, "tickfont": {"size": 20, 'family':
                            'latex'}}, 'yaxis': {'title': {"text": 'PCo2', "font": {"size": 25, 'family': 'latex'}},
                            "tickfont": {"size": 20, 'family': 'latex'}, 'zeroline': False, 'scaleratio': 1,
                            'scaleanchor': 'x', 'showgrid': False, "showline": True, "linewidth": 2}, 'legend':
                            {'x': 0, 'y': 1, "font": {"size": 17.5, 'family': 'latex'}}, "width": 500,
                            "height": 500}})], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='DOC - Control',
                    figure={'data': fig_control_asd, 'layout': go.Layout(xaxis={'title': {"text": 'Overlap', "font":
                           {"size": 25, 'family': 'latex'}}, 'range': [0.95, np.max(new_overlap_vector)],
                            'scaleratio':1, 'showgrid': False, "automargin": True, "showline": True, "linewidth": 2},
                            yaxis={'title': {"text": 'Dissimilarity', "font": {"size": 25, 'family': 'latex'}}, 'range':
                            [0.25, 0.6], 'scaleratio':1, 'showgrid': False, "automargin": True, "showline": True,
                            "linewidth":2}, xaxis_tickfont=dict(size=20, family='latex'), yaxis_tickfont=dict(size=20,
                            family='latex'), width=500, height=500)})], style={'width': '48%', 'display':
                            'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='DOC - Control only',
                    figure={'data': fig_control, 'layout': go.Layout(xaxis={'title': {"text": 'Overlap', "font":
                            {"size": 25, "family": "latex"}}, 'range': [0.74, 1], 'scaleratio':1,
                            'showgrid': False, "automargin": True, "showline": True, "linewidth": 2},
                            yaxis={'title': {"text": 'Dissimilarity', "font": {"size": 25, "family": "latex"}},
                            'range': [0, 0.7], 'scaleratio':1, 'showgrid': False, "automargin": True, "showline": True,
                            "linewidth":2}, xaxis_tickfont=dict(size=20), yaxis_tickfont=dict(size=20), width=500,
                             height=500)})], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(
            id='DOC - ASD',
            figure={'data': fig_ASD, 'layout': go.Layout(xaxis={'title': {"text": 'Overlap', "font":{"size": 25,
                    "family": "latex"}}, 'range': [0.74, 1], 'scaleratio': 1,
                    'showgrid': False, "automargin": True, "showline": True, "linewidth": 2}, yaxis={'title': {"text":
                    'Dissimilarity', "font": {"size": 25, "family": "latex"}}, 'range': [0, 0.7], 'scaleratio': 1,
                    'showgrid': False, "automargin": True, "showline": True, "linewidth":2}, xaxis_tickfont=dict(
                    size=20), yaxis_tickfont=dict(size=20), width=500, height=500)})], style={'width': '48%', 'display':
                    'inline-block'})
])

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8080)