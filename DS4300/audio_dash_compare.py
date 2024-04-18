from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import plotly.express as px
import pandas as pd
import networkx as nx
import math
import audio_features as af
import audio_sort
import numpy as np
from dash.exceptions import PreventUpdate
from neomodel_graph import Sample, connect_similar, connect_different, exec_query
from sklearn.metrics.pairwise import cosine_similarity

# dataframe for samples
dt_data = pd.DataFrame(columns=['Sample Name',
                                'Length',
                                'Attack',
                                'Release',
                                 'Bass',
                                'Mids',
                                'Treble',
                                'Air',
                                'Group'])

# dataframe for recommendations
rec_df = pd.DataFrame(columns= [
                                'Sample Name',
                                'Length',
                                'Attack',
                                'Release',
                                'Bass',
                                'Mids',
                                'Treble',
                                'Air',
                                'Group',
                                'Type'])


# build dashboard
app = Dash('Audio Visualizer')
app.layout = html.Div([
        dcc.Store(id='uploaded_data', storage_type='session'),
        dcc.Store(id='uploaded_data_nodes', storage_type='session'),


        dcc.Tabs(id="vis_tabs", value='compare_tab', children=[
            dcc.Tab(label='Visualizations', value='compare_tab'),
            dcc.Tab(label='Graph Database', value='groups_tab'),
            dcc.Tab(label='Data Table', value='datatable_tab'),
        ]),
        html.Div(id='tab_content'),



])


@callback(
    Output("tab_content", "children"),
    Input('vis_tabs', 'value'))
def render_content(tab):
    # page 1
    if tab == 'compare_tab':
        return html.Div([
            html.Div([
                # upload sample
                dcc.Upload(
                    id='upload-soundfile',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select File')
                    ]),
                    style={
                        "width": "80%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "10px",
                    },
                ),
                # waveform graphs
                dcc.Graph(id='waveform', style={'width': '500px', 'height': '50vh'}, figure={}),

                # BER graphs (overall)
                dcc.Graph(id='ber', style={'width': '500px', 'height': '50vh'}, figure={}),

                # Levels of B, LM, HM, T
                dcc.Graph(id='levels', style={'width': '500px', 'height': '50vh'}, figure={}),

                # RMS graphs
                dcc.Graph(id='rms_e', style={'width': '500px', 'height': '50vh'}, figure={}),

            ], style={'flex': '1',
                      'display': 'inline-block',
                      'width': '25%'}),

            html.Div([
                # upload sample #2
                dcc.Upload(
                    id='upload-soundfile2',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select File')
                    ]),
                    style={
                        "width": "80%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "10px",
                    },
                ),

                # waveform graphs
                dcc.Graph(id='waveform2', style={'width': '500px', 'height': '50vh'}, figure={}),

                # BER graphs (overall)
                dcc.Graph(id='ber2', style={'width': '500px', 'height': '50vh'}, figure={}),

                # Levels of B, LM, HM, T
                dcc.Graph(id='levels2', style={'width': '500px', 'height': '50vh'}, figure={}),

                # RMS graphs
                dcc.Graph(id='rms_e2', style={'width': '500px', 'height': '50vh'}, figure={}),

            ], style={'flex': '1',
                      'display': 'inline-block',
                      'width': '25%'}),

            # options sidebar

            html.Div([
                # BER graph threshold slider (for both)
                html.H3('Frequency of Comparison (for Band Energy Ratio Graph)'),
                dcc.Input(id='ber_threshold', type='number', value=2000),

                # Combine/Compare RMS graphs
                html.H3('RMS Graph Options'),
                dcc.RadioItems(['RMS', 'Amplitude Envelope', 'Both'], 'RMS',
                               id='rms_graph_options'),

            ], style={
                      'display': 'inline-block',
                      'width': '25%'}),


        ], style={'display': 'flex'})
    elif tab == 'groups_tab':
        return html.Div([
            # rec
            html.Div([
                dash_table.DataTable(rec_df.to_dict('records'),
                                     [{"name": i, "id": i} for i in rec_df.columns],
                                     id='focus_sample_info',
                                     editable=False,
                                     row_deletable=False),
            ], style={'flex': '1',
                      'display': 'inline-block',
                      'width': '70%'}),

            # SIDEBAR
            html.Div([

                html.H3('Filters:'),
                html.P('Use the sliders below to filter the samples uploaded before getting recommendations'),

                html.H4('Brightness (Percentile)'),
                dcc.Slider(0, 100, 10, value=70, id='brightness_percentile'),

                html.H4('Attack (Percentile)'),
                dcc.Slider(0, 100, 10, value=70, id='attack_percentile'),

                html.H4('Release (Percentile)'),
                dcc.Slider(0, 100, 10, value=70, id='release_percentile'),

                html.H4('Length (Percentile)'),
                dcc.Slider(0, 100, 10, value=70, id='length_percentile'),

                html.H3('Show Individual Recommendations for:'),
                dcc.Dropdown(['-none-'],
                             value='-none-',
                             id='indiv_connect'),

                html.Button(id='initiate_recommend', children=['Recommend!'], n_clicks=0)

            ], style={'flex': '1',
                      'display': 'inline-block',
                      'width': '20%'})
        ])
    elif tab == 'datatable_tab':
        return html.Div([
            html.Div([
                # upload samples
                dcc.Upload(
                    id='upload-soundfile_dt',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        "width": "95%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "10px",
                    },
                    multiple=True
                )]),

            dash_table.DataTable(dt_data.to_dict('records'),
                                 [{"name": i, "id": i} for i in dt_data.columns],
                                 id='data-table',
                                 editable=False,
                                 row_deletable=True,
                                 filter_action='native'),
        ])

@app.callback(
    [Output('waveform', 'figure'),
     Output('ber', 'figure'),
     Output('levels', 'figure'),
     Output('rms_e', 'figure')],
    Input('upload-soundfile', 'contents'),
    Input('ber_threshold', 'value'),
    Input('rms_graph_options', 'value'),
    State(component_id='upload-soundfile', component_property='filename')
)
def figs(soundfile, ber_threshold, rms_graph_options, filename):

    signal, sr, t = af.load_audio(soundfile)

    # waveforms
    fig1wav = px.line(x=t,
                      y=signal,
                      title=f'Waveform: {filename}',
                      labels={'x': "Frame Number", 'y': 'Amplitude'},
                      range_y=[-1, 1],
                      width=500,
                      height=400)

    # ber
    ber, t_ber = af.band_energy_ratio(signal, ber_threshold)

    fig1ber = px.line(x=t_ber,
                      y=ber,
                      title=f'Band Energy Ratio',
                      labels={'x': "Time(s)", 'y': 'Amplitude'},
                      width=600,
                      height=400)

    # levels
    bass, mids, treble, air = af.levels(signal, sr)
    levels_df = pd.DataFrame({'Frequency Range': ['Bass', 'Mids', 'Treble', 'Air'],
                              'Energy': [bass, mids, treble, air]})
    fig1lev = px.bar(levels_df,
                     title='Frequency Range Levels',
                     x='Frequency Range',
                     y='Energy')

    # energy
    ae, t_ae, rms, t_rms = af.signal_energy(signal)

    # Amplitude Envelope
    energy_df = pd.DataFrame({'time': t_ae, 'ae':  ae, 'rms': rms[0]})

    fig_nrg = 0
    if rms_graph_options == 'RMS':
        fig_nrg = px.line(x=t_rms,
                          y=rms[0],
                          title='Root Mean Square Energy',
                          labels={'x': "Time(s)", 'y': 'Root Mean Square Energy'},
                          width=600,
                          height=400)
    elif rms_graph_options == 'Amplitude Envelope':
        fig_nrg = px.line(x=t_ae,
                         y=ae,
                         title='Amplitude Envelope',
                         labels={'x': "Time(s)", 'y': 'Maximum Amplitude'},
                         width=600,
                         height=400)
    else:
        fig_nrg = px.line(energy_df,
                          x='time',
                          y=energy_df.columns[1:],
                          width=600,
                          height=400,
                          title='Energy (AE + RMS)'
                          )

    return fig1wav, fig1ber, fig1lev, fig_nrg

@app.callback(
    [Output('waveform2', 'figure'),
     Output('ber2', 'figure'),
     Output('levels2', 'figure'),
     Output('rms_e2', 'figure')],
    Input('upload-soundfile2', 'contents'),
    Input('ber_threshold', 'value'),
    Input('rms_graph_options', 'value'),
    State(component_id='upload-soundfile2', component_property='filename')
)
def figs2(soundfile2, ber_threshold, rms_graph_options, filename2):
    signal, sr, t = af.load_audio(soundfile2)

    # waveforms
    fig2wav = px.line(x=t,
                      y=signal,
                      title=f'Waveform: {filename2}',
                      labels={'x': "Frame Number", 'y': 'Amplitude'},
                      range_y=[-1, 1],
                      width=500,
                      height=400)

    # ber
    ber2, t_ber2 = af.band_energy_ratio(signal, ber_threshold)

    fig2ber = px.line(x=t_ber2,
                      y=ber2,
                      title=f'Band Energy Ratio',
                      labels={'x': "Time(s)", 'y': 'Amplitude'},
                      width=600,
                      height=400)

    # levels
    bass, mids, treble, air = af.levels(signal, sr)
    levels_df = pd.DataFrame({'Frequency Range': ['Bass', 'Mids', 'Treble', 'Air'],
                              'Energy': [bass, mids, treble, air]})
    fig2lev = px.bar(levels_df,
                     x='Frequency Range',
                     y='Energy')

    # energy
    ae2, t_ae2, rms2, t_rms2 = af.signal_energy(signal)

    # Amplitude Envelope
    energy_df = pd.DataFrame({'time': t_ae2, 'ae': ae2, 'rms': rms2[0]})

    if rms_graph_options == 'RMS':
        fig_nrg = px.line(x=t_rms2,
                          y=rms2[0],
                          title='Root Mean Square Energy',
                          labels={'x': "Time(s)", 'y': 'Root Mean Square Energy'},
                          width=600,
                          height=400)
    elif rms_graph_options == 'Amplitude Envelope':
        fig_nrg = px.line(x=t_ae2,
                          y=ae2,
                          title='Amplitude Envelope',
                          labels={'x': "Time(s)", 'y': 'Maximum Amplitude'},
                          width=600,
                          height=400)
    else:
        fig_nrg = px.line(energy_df,
                          x='time',
                          y=energy_df.columns[1:],
                          width=600,
                          height=400,
                          title='Energy (AE + RMS)'
                          )
    return fig2wav, fig2ber, fig2lev, fig_nrg

@app.callback(
    Output('data-table', 'data'),
    # Input('vis_tabs', 'value'),
    Input('upload-soundfile_dt', 'contents'),
    State('upload-soundfile_dt', 'filename'),
    State('uploaded_data', 'data'),
)
def update_datatable(files, filenames, existing_data):
    # print(f'Existing:{existing_data}')
    if existing_data is not None:
        if files is None:
            return existing_data
        else:
            pass
    else:
        pass

    # data features
    signal_data = [af.load_audio(file) for file in files]  #
    signal_envs = [af.signal_energy(signal[0])[0] for signal in signal_data]
    total = len(signal_data)
    attacks_releases = [[af.attack_release(signal_data[i], signal_envs[i])] for i in range(total)]
    levels = [af.levels(signal_data[i][0], 48000) for i in range(total)]
    clusters = math.ceil(math.sqrt(len(files)))

    all_sample_data = [signal_data[i][0] for i in range(total)]
    timbres = audio_sort.KMeans_init(all_sample_data, clusters)
    stuff = {'Sample Name': filenames,
             'Length': [len(i[0]) / 48000 for i in signal_data],
             'Attack': [i[0][0] for i in attacks_releases],
             'Release': [i[0][1] for i in attacks_releases],
             'Bass': [i[0] for i in levels],
             'Mids': [i[1] for i in levels],
             'Treble': [i[2] for i in levels],
             'Air': [i[3] for i in levels],
             'Group': timbres}

    stuff2 = [{key: stuff[key][i] for key in stuff.keys()} for i in range(total)]

    stuff_df = pd.DataFrame(stuff2)
    list_form = (stuff_df.to_dict('records'))

    # temporary, flawed method (no retraining of the model)
    if existing_data:
        list_form.extend(existing_data)
        return list_form
    else:
        return stuff_df.to_dict('records')



# re-update datatable
@app.callback(
    Output('uploaded_data', 'data'),
    Input('data-table', 'data')
)
def datatable_to_store(data):
    if data is None:
        # print("No Data.")
        raise PreventUpdate
    else:
        print(f'Adding data...:')
        return data


@app.callback(
    Output('data-table', 'data', allow_duplicate=True),
    Input('uploaded_data', 'data'),
    Input('vis_tabs', 'value'),
    prevent_initial_call=True,
)
def store_to_datatable(existing_data, tab):
    if existing_data is None:
        print('No data.')

    if tab != 'datatable_tab':
        raise PreventUpdate
    elif tab == 'datatable_tab' and existing_data is not None:
        print(f'Yes Data')
        return existing_data
    else:
        print('No data.')


@app.callback(
    Output('focus_sample_info', 'data'),
    Input('uploaded_data', 'data'),
    Input('indiv_connect', 'value'),
    Input('indiv_connect', 'options'),
    State('initiate_recommend', 'n_clicks'),
    prevent_initial_call=True
)
def rec_table(data, focus_sample, all_choices, n_clicks):

    if n_clicks is None:
        pass
    else:
        rec_close, rec_far = audio_sort.edge_distance(data)

        # decide edges
        samples = [Sample(id=idx,
                          name=sample_data['Sample Name'],
                          length=sample_data['Length'],
                          atk=sample_data['Attack'],
                          rel=sample_data['Release'],
                          bass=sample_data['Bass'],
                          mids=sample_data['Mids'],
                          treble=sample_data['Treble'],
                          air=sample_data['Air'],
                          group=sample_data['Group']) for idx, sample_data in enumerate(data)]
        for i in rec_close:
            connect_similar(samples[i[0]], samples[i[1]], i[2])
        for j in rec_far:
            connect_different(samples[j[0]], samples[j[1]], j[2])
        query = (
                f"MATCH (n:MyNode)-[r:ConnectsTo]->(m) WHERE n.name = '{data.index(focus_sample)}' RETURN n, r, m"
                )
        query_result = exec_query(query)
        print(query_result)


def rec_func(data):
    pass

@app.callback(
    Output('indiv_connect', 'options'),
    Input('uploaded_data', 'data')
)
def update_indiv_select(data):

    filenames = [i['Sample Name'] for i in data]

    return filenames


def dash_start():
    app.run_server(debug=True)


if __name__ == "__main__":
    dash_start()
