
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.subplots as ps
import librosa
import librosa.display
import numpy as np
import base64
import io
import scipy as sp
import math


def main():
    # insert dash app stuff idk

    # app object:
    app = Dash('Audio Visualizer')

    # app layout
    app.layout = html.Div([
        html.H1('Audio Data Visualizer'),
        html.P('This is a basic tool used to visualize different aspects of an audio sample, given a .wav file.'
               ' The aspects displayed are: the waveform shape, the Band Energy Ratio (ratio of energy below a '
               ' frequency to energy above it), the Zero-Crossing Rate (# of times the signal crosses zero in a'
               ' frame of 512 samples), the results of applying the Fast Fourier Transform on the signal'
               '(Fundamental frequencies and their magnitudes), and two ways to measure the energy in a signal'
               ' (Amplitude Envelope and Root Mean Square Energy with a frame size of 512 samples).  I had planned '
               'to add spectrograms and visualizations of MFCC components as well, as they are commonly used'
               " in machine learning, but I couldn't get them working in time."),
        html.H4('*Note that larger files may take a longer time to generate visualizations for'),

        # upload
        dcc.Upload(
            id='upload-soundfile',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },

        ),

        # graph
        dcc.Graph(id='audio_upload', style={'width': '1400px', 'height': '90vh'}, figure={}),

        # sliders
        html.H3('Frequency of Comparison (for Band Energy Ratio Graph)'),
        dcc.Slider(100, 10000, 300, value=900, id='div_freq'),

        html.H3('Frequency-Magnitude Graph Zoom'),
        dcc.Slider(0.5, 3, 0.1, value=1, id='zoom')

    ])

    def band_energy_ratio(signal, div_freq):
        """Calculates the band energy ratio at each timeframe in the signal
        :param signal: input signal (array)
        :param div_freq: threshold frequency for comparison (int/float)
        :return: Band energy ratio (below/above)
        """

        # get spectrogram data
        spectrogram_data = librosa.stft(signal, n_fft=512, hop_length=256)

        # Get split frequencies
        # shape of the spectrogram = (# of frequency bins, time)
        freq_gap = 24000 / spectrogram_data.shape[0]
        divider_bin = int(np.floor(div_freq / freq_gap))

        # Calculate BER
        power_spectrogram = np.abs(spectrogram_data) ** 2
        # transpose to get (time, frequency bins) to iterate through the list of frequency bins at each time frame
        power_spectrogram = power_spectrogram.T
        ber = np.array([np.sum(frame[:divider_bin]) / np.sum(frame[divider_bin:]) for frame in power_spectrogram])

        # clean infinity values
        ber[ber == np.inf] = np.nan
        ber = np.nan_to_num(ber)
        return ber

    @app.callback(
        Output(component_id='audio_upload', component_property='figure'),
        Input(component_id='upload-soundfile', component_property='contents'),
        Input(component_id='div_freq', component_property='value'),
        Input(component_id='zoom', component_property='value'),
        State(component_id='upload-soundfile', component_property='filename')
    )
    def display_waveform(upload, div_freq, zoom, filename):
        """
        generates all graphs for the uploaded data
        :param upload: contents of uploaded file
        :param div_freq: slider value to adjust the dividing frequency on the BER graph
        :param zoom: slider value to adjust how zoomed in the Fourier Transform graph is about the origin.
        :param filename: name of uploaded file
        :return: grid of 6 graphs for waveform, band-energy ratio, zero-crossing rate, fourier-transform,
        amplitude envelope, and rms energy
        """

        # extract contents from the uploaded file
        content_type, content_string = upload.split(',')

        # decode the file content
        decoded = io.BytesIO(base64.b64decode(content_string))

        # get signal data and sample rate from decoded content
        signal, sr = librosa.load(decoded, sr=None)

        # get data I want to present
        ber = band_energy_ratio(signal, div_freq)
        zcr = librosa.feature.zero_crossing_rate(signal, frame_length=512, hop_length=256)

        # fourier
        fourier = sp.fft.fft(signal)
        mag = np.absolute(fourier)

        # set frequency range and bounds for the fourier-transform results (excludes extraneous info past 24000hz)
        freq_range = np.linspace(0, 24000, len(mag))
        bound = int(math.floor(24000/zoom))

        # energy analysis data
        ae = np.array([max(signal[i:i + 512]) for i in range(0, len(signal), 256)])
        rms = librosa.feature.rms(y=signal, frame_length=512, hop_length=256)

        # set x_arrays to use for x-axes of graphs
        t = librosa.frames_to_time(range(signal.size), hop_length=256)
        t_ber = librosa.frames_to_time(range(ber.size), hop_length=256)
        t_zcr = librosa.frames_to_time(range(zcr.size), hop_length=256)
        t_ae = librosa.frames_to_time(range(ae.size), hop_length=256)
        t_rms = librosa.frames_to_time(range(rms.size), hop_length=256)

        # create subplots
        fig = ps.make_subplots(rows=3,
                               cols=2,
                               horizontal_spacing=0.1,
                               vertical_spacing=0.2,
                               subplot_titles=(f'Waveform of {filename}',
                                               'Band Energy Ratio',
                                               'Zero-Crossing Rate Per 256 Samples',
                                               'Magnitudes on Frequency Spectrum',
                                               'Energy Analysis: Amplitude Envelope',
                                               'Energy Analysis: RMS Energy'),
                               )
        # Visualization of Waveform
        fig_wav_vis = px.line(x=t,
                              y=signal,
                              title=f'Waveform',
                              labels={'x': "Sample Number", 'y': 'Amplitude'},
                              range_y=[-1, 1],
                              width=800,
                              height=800)
        # BER
        fig_ber = px.line(x=t_ber,
                          y=ber,
                          title=f'Band Energy Ratio',
                          width=800,
                          height=800)

        # Zero Crossing Rate
        fig_zcr = px.line(x=t_zcr,
                          y=zcr[0],
                          title='Zero-Crossing Rate Per 256 Samples',
                          width=800,
                          height=800)

        # Fourier Transform
        fig_fourier = px.line(x=freq_range[:bound],
                              y=mag[:bound],
                              title='Magnitudes of Signal at Frequencies',
                              width=800,
                              height=800)
        # Amplitude Envelope
        fig_ae = px.line(x=t_ae,
                         y=ae,
                         title='Amplitude Envelope',
                         width=800,
                         height=800)
        # RMS Energy
        fig_rms = px.line(x=t_rms,
                          y=rms[0],
                          title='Root Mean Square Energy',
                          width=800,
                          height=800)

        def append_trace(figure):
            """

            :param figure: figure
            :return: Traces to use to apply information to each subplot slot
            """
            fig_traces = []
            for trace in range(len(figure['data'])):
                fig_traces.append(figure['data'][trace])
            return fig_traces

        # arrangement of figures in order
        fig_list = [fig_wav_vis, fig_ber, fig_zcr, fig_fourier, fig_ae, fig_rms]
        all_traces = np.array([append_trace(fig) for fig in fig_list]).reshape(-1, 2)

        # add each trace to their respective slot
        for row_n in range(len(all_traces)):
            for traces_n in range(len(all_traces[row_n])):
                fig.add_trace(all_traces[row_n][traces_n], row=row_n+1, col=traces_n+1)

        # renaming axes
        fig.update_xaxes(title='Frame-Number', row=1, col=1)
        fig.update_yaxes(title='Amplitude', row=1, col=1)

        fig.update_xaxes(title='Time(s)', row=1, col=2)
        fig.update_yaxes(title='Below:Above Ratio', row=1, col=2)

        fig.update_xaxes(title='Time(s)', row=2, col=1)
        fig.update_yaxes(title='Zero-Crossing Rate', row=2, col=1)

        fig.update_xaxes(title='Frequency(Hz)', row=2, col=2)
        fig.update_yaxes(title='Magnitude', row=2, col=2)

        fig.update_xaxes(title='Time(s)', row=3, col=1)
        fig.update_yaxes(title='Highest Amplitude', row=3, col=1)

        fig.update_xaxes(title='Time(s)', row=3, col=2)
        fig.update_yaxes(title='RMS Energy', row=3, col=2)

        return fig

    app.run_server(debug=True)


if __name__ == '__main__':
    main()
