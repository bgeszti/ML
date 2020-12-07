import plotly_express as px
import dash
import dash_html_components as html
import pandas as pd
from plotly import tools
import plotly
import dash_core_components as dcc
import plotly.graph_objs as go
from dash.dependencies import Input, Output

mapbox_access_token = 'pk.eyJ1IjoidG9sbmFpeiIsImEiOiJjajVqdnJxdTIyaG45MzJvZGg5N3JjeWk3In0.npvpLSXfhsmCy0xFOrjxHA'

# Import datasets
df1 = pd.read_csv('data/SFBay.csv')
df2 = pd.read_csv('data/stations.csv')

# merge stations with measurements
df1 = pd.merge(df1, df2[['Station.Number','Depth MLW']], on='Station.Number')

# select the required fields from df1
df1=df1[[
    'TimeStamp',
    'Temperature',
    'Salinity',
    'Chlorophyll',
    'Oxygen.Electrode',
    'Oxygen.Saturation',
    'Calculated.Oxygen',
    'Optical.Backscatter',
    'Calculated.SPM',
    'Sigma.t',
    'Station.Number',
    'Depth MLW'
]]

# convert timestamp strings to datetime
df1['TimeStamp'] = pd.to_datetime(df1['TimeStamp'])

df1['year'] = df1['TimeStamp'].dt.year

# convert DMS to DD for location data
def dms_to_dd(d, m, s):
    sign = -1 if d<0 else 1
    dd = abs(d) + float(m)/60 + float(s)/3600
    return dd * sign

# convert location data from DMS to DD
df2['Latitude']=df2.apply(lambda x: dms_to_dd(x['LatD'],x['LatM'],x['LatS']), axis=1)
df2['Longitude']=df2.apply(lambda x: dms_to_dd(x['LongD'],x['LongM'],x['LongS']), axis=1)

# drop redundant location data
df2.drop(['LatD','LatM','LatS','LongD','LongM','LongS'], axis=1)

measurements = df1.columns[1:10]

# list of possible aggregations in the selectbox
aggregations = [
    {'label': 'Monthly', 'value':'M'},
    {'label': 'Quarterly', 'value':'Q'},
    {'label': 'Yearly', 'value':'Y'},
]

# set minimum and maximum value for range-year slider
min_year = 2000
max_year = 2010

# set plot colors
colors1=[[0, '#1a2543'], [0.3, '#474049'], [0.5, '#a27755'],[0.7, '#cf925b'],[1, '#FDAE61']]
colors2=[[0, '#1a2543'], [0.3, '#1d365a'], [0.6, '#245888'],[0.8, '#28699f'],[1, '#2C7BB6']]

# make list of possible measurements for the measurements select box
OPTIONS = [{'label': i.title(), 'value': i} for i in measurements]

# collect unique depth values for depth slider
unique_depths = list(df2['Depth MLW'].unique())
unique_depths.sort()

# initialize dash app with some external styling
app = dash.Dash(__name__)

# build main title
def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.H6("Weather Records for Seattle")
        ],
    )

# build subtitle
def build_graph_title(title, style={}):
    return html.P(className="graph-title", children=title, style=style)

# app layout
app.layout = html.Div(
    children=[
        html.Div(
            id="top-row",
            children=[
                html.Div(
                    className="row",
                    id="top-row-header",
                    children=[
                        html.Div(
                            id="header-container",
                            children=[
                                build_banner(),
                                html.P(
                                    id="instructions",
                                    children=(
                                        "Use the dropdown boxes and sliders to select measurements (max. 2 at the same time) and time or depth ranges. "
                                        "Select data points from the map to visualize data for the selected station(s). "
                                        "Selection could be done by clicking on individual data points or using the lasso tool to capture multiple data points. "
                                        "Use the graph's menu bar to select, zoom or save the plots."
                                    )
                                ),
                                html.Div(
                                    className="row",
                                    style={"marginBottom":"30px"},
                                    children=[
                                        html.Div(
                                            className="six columns",
                                            children=[
                                                build_graph_title("Measurements"),
                                                dcc.Dropdown(multi=True, clearable=False, value=['Temperature'],id = 'opt', options = OPTIONS)
                                            ]
                                        ),
                                        html.Div(
                                            className="six columns",
                                            children=[
                                                build_graph_title("Aggregation"),
                                                dcc.Dropdown(multi=False, value='Q',id = 'aggregation', options = aggregations)
                                            ]
                                        )
                                    ]
                                ),
                                build_graph_title("Time Period"),
                                dcc.RangeSlider(
                                    id="year-range", 
                                    count=1, 
                                    min=min_year, 
                                    max=max_year, 
                                    step=1, 
                                    value=[2000, 2010],
                                    marks={str(y) : {'label' : str(y), 'style':{'color':"white"}} for y in range(2000, 2011)}
                                ),
                            ],
                        )
                    ],
                ),
                html.Div(
                    className="row",
                    id="top-row-graphs",
                    children=[
                        # map
                        html.Div(
                            id="map-container",
                            className="five columns",
                            children=[
                                build_graph_title("Map"),
                                dcc.Graph(id='map', animate=True)
                            ],
                        ),
                        # scatter plot
                        html.Div(
                            id="scatter-container",
                            className="seven columns",
                            children=[
                                html.Div(
                                    id="ternary-header",
                                    children=[
                                        build_graph_title(
                                            "Measurements on scatter graph"
                                        ),
                                    ],
                                ),
                                dcc.Loading(
                                    type="cube",
                                    style={'backgroundColor': '#f2f5fa', 'height': '450px', 'display': 'flex', 'alignItems': 'center'},
                                    children=dcc.Graph(id="scatter")
                                )
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="row",
            id="bottom-row",
            children=[
                # contour plots
                html.Div(
                    id="contour-plot-container",
                    className="six columns",
                    children=[
                        build_graph_title("Measurements on contour plots"),
                        dcc.Loading(type="cube", style={'backgroundColor': '#f2f5fa', 'height': '450px', 'display': 'flex', 'alignItems': 'center'}, 
                        children=dcc.Graph(id="depth-graph")),
                        build_graph_title("Depth MLW", {'marginLeft':'25px', 'marginTop':'10px'}),
                        dcc.RangeSlider(
                            id="depth-range", 
                            step=None, 
                            marks={idx : {'label' : str(y), 'style':{'color':"black", 'fontSize':"10px"}} for idx,y in enumerate(unique_depths)}, 
                            min=0, 
                            max=len(unique_depths)-1, 
                            value=[0, len(unique_depths)-1] 
                        )
                    ],
                ),
                html.Div(
                    # heatmap plots
                    id="heatmap-container",
                    className="six columns",
                    children=[
                        build_graph_title("Measurements on heatmaps"),
                        dcc.Loading(
                            type="cube", 
                            style={'backgroundColor': '#f2f5fa', 'height': '450px', 'display': 'flex', 'alignItems': 'center'},
                            children=dcc.Graph(id="heatmap")
                        )
                    ],
                ),
            ],
        ),
    ]
)
# update all plots except contour plots
@app.callback(
    [Output('scatter', 'figure'), Output('map', 'figure'), Output('heatmap','figure')],
    [Input('opt', 'value'), Input('year-range', 'value'), Input('map', 'selectedData'), Input('aggregation','value')])
def update_figure(selected,year,clickData, aggregation):

    # select temperature if no measurement is selected
    if len(selected) == 0:
        selected = ['Temperature']

    # select one or more years from the dataset
    if year[0] != year[1]:
        tdf = df1[(df1['year'] >= year[0]) & (df1['year'] <= year[1])]
    else :
        tdf = df1[df1['year'] == year[0]]

    # select stations if there are selections on the map
    if(clickData):
        row_indexes = list(map(lambda x: int(x['pointIndex']), clickData['points']))
        selected_station_numbers = df2.iloc[row_indexes]['Station.Number']
        dff = tdf[tdf['Station.Number'].isin(selected_station_numbers)]
    else:
        dff=tdf

    # index the dataset by timestamp
    dff.index = dff['TimeStamp']

    # resample data with selected aggregation
    dff_r=dff.resample(aggregation).mean()
    
    # make scatter plot
    fig = plotly.subplots.make_subplots(
        rows=len(selected), 
        cols=1,
        shared_xaxes=True, 
        shared_yaxes=False,
        vertical_spacing=0.1
    )
    trace = []
    fig.update_xaxes(row=len(selected), col=1)  
    
    for i in range(len(selected)):
        fig.update_yaxes(title_text=selected[i], row=i+1, col=1)
        trace.append(go.Scatter(
            x=dff_r.index, 
            y=dff_r[selected[i]], 
            name=selected[i], 
            mode='lines+markers', 
            connectgaps=True,
            marker={'size': 10, "opacity": 1, "line": {'width': 1}}
        ))
        fig.append_trace(trace[i], i+1, 1)
        
    fig['layout'].update(
        colorway=['#fdae61', '#800080', '#2c7bb6'], 
        margin=dict(l=50, r=20, t=30, b=40), 
        showlegend=False, 
        xaxis_tickformatstops = [
            dict(dtickrange=["M1", "M12"], value="%b %Y"),
            dict(dtickrange=["M12", None], value="%Y")
        ],
        paper_bgcolor='#f2f5fa'
    )

    # make mapbox
    mapbox = go.Figure(go.Scattermapbox(
        lat=df2["Latitude"], 
        lon=df2["Longitude"],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=10,
            color='#800080'
        ),
        text=df2["Location"],
        hoverinfo='text'
    ))

    mapbox.update_layout(
        clickmode='event+select',
        margin=dict(l=0, r=0, t=0, b=0),
        autosize=True,
        hovermode='closest',
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=37.82,
                lon=-122.26
            ),
            pitch=0,
            zoom=8,
            style='light'
        ),
    )

    # make heatmap
    fig2 = plotly.subplots.make_subplots(
        rows=len(selected),
        cols=1,
        shared_xaxes=True,
        shared_yaxes=False,
        vertical_spacing=0.1
    )
    fig2.update_xaxes(row=len(selected), col=1)
    fig2.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    fig2.update_layout(
        xaxis_tickformatstops = [
            dict(dtickrange=["M1", "M12"], value="%b %Y"),
            dict(dtickrange=["M12", None], value="%Y"),
        ],
        paper_bgcolor='#E5ECF6'
    )
    trace2 = []

    colors=[colors1,colors2]
    for i in range(len(selected)):
        fig2.update_yaxes(row=i+1, col=1, tickangle=270, title_text=selected[i])
        trace2.append(go.Heatmap(
            x=dff_r.index,
            y=[''],
            z=[dff_r[selected[i]]],
            xgap = 0,
            ygap = 0,
            connectgaps=True,
            colorscale=colors[i],
            showscale=False
        ))
        fig2.append_trace(trace2[i], i+1, 1)

    # return with the three generated plots
    return [
        fig, 
        mapbox, 
        fig2
    ]

# update the contour plots
@app.callback(
    [Output('depth-graph', 'figure')],
    [Input('opt', 'value'), Input('year-range', 'value'), Input('aggregation','value'), Input('depth-range','value')])
def update_figure(selected,year, aggregation, depth_range):

    # select temperature if no measurement is selected
    if len(selected) == 0:
        selected = ['Temperature']

    # filter for one or more year from the measurements
    if year[0] != year[1]:
        tdf = df1[(df1['year'] >= year[0]) & (df1['year'] <= year[1])]
    else :
        tdf = df1[df1['year'] == year[0]]

    # filter the selected depth range if there is a range selected
    if depth_range[0] != depth_range[1]:
        tdf = tdf[(tdf['Depth MLW'] >= unique_depths[depth_range[0]]) & (tdf['Depth MLW'] <= unique_depths[depth_range[1]])]
    else :
        tdf = tdf

    # aggregate the measurements and get rid of double indexing
    depth_data = tdf
    depth_data.index = depth_data['TimeStamp']
    depth_data = depth_data.groupby(['Depth MLW',pd.Grouper(freq=aggregation)]).mean()
    depth_data_reset = depth_data.reset_index()
    depth_data_reset.columns = [
        'Depth MLW', 
        'TimeStamp', 
        'Temperature', 
        'Salinity',
        'Chlorophyll',
        'Oxygen.Electrode',
        'Oxygen.Saturation',
        'Calculated.Oxygen',
        'Optical.Backscatter',
        'Calculated.SPM',
        'Sigma.t',
        'Station.Number',
        'year'
    ]
    
    # make cotour plot
    depths = plotly.subplots.make_subplots(
        rows=len(selected),
        cols=1,
        shared_xaxes=True, 
        shared_yaxes=False,
        vertical_spacing=0.1
    )
    depth_traces = []
    colors=[colors1,colors2]

    for i in range(len(selected)):
        depths.update_yaxes(title_text=selected[i], row=i+1, col=1)
        depth_traces.append(go.Contour(
            z=depth_data_reset[selected[i]],
            y=depth_data_reset['Depth MLW'],
            x=depth_data_reset['TimeStamp'], 
            showscale=False,
            contours_coloring='heatmap',
            connectgaps=True,colorscale=colors[i]
        ))
        depths.append_trace(depth_traces[i], i+1, 1)

    # reversed y axis for depth data
    depths.update_yaxes(autorange="reversed")
    
    depths.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='#E5ECF6'
    )

    # return with contour plot
    return [depths]

# this callback controls the maximum selectable measurements
@app.callback(
    Output(component_id="opt", component_property="options"),
    [Input(component_id="opt", component_property="value")],
)
def update_dropdown_options(values):
    if len(values) == 2:
        return [option for option in OPTIONS if option["value"] in values]
    else:
        return OPTIONS

# start the server if we are running the script by hand
if __name__ == '__main__':
    app.run_server(debug=True)

# pass the server instance for gunicorn in production
server = app.server
