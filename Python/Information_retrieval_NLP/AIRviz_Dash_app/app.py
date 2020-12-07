import plotly_express as px
import dash
import dash_html_components as html
import pandas as pd
from plotly import tools
import plotly
import dash_core_components as dcc
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import textwrap
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import matplotlib
import re

import time

matplotlib.use("Agg")


# Import validation results datasets
knrm_val = pd.read_csv('data/knrm_validationResults.csv',index_col=0)
knrm_loss = pd.read_csv('data/knrm_lossResults.csv',index_col=0)
knrm_loss = knrm_loss.rename(columns={"0": "Loss"})
mp_val = pd.read_csv('data/match_pyramid_validationResults.csv', index_col=0)
mp_loss = pd.read_csv('data/match_pyramid_lossResults.csv',index_col=0)
mp_loss = mp_loss.rename(columns={"0": "Loss"})
conv_knrm_val = pd.read_csv('data/conv_knrm_validationResults.csv',index_col=0)
conv_knrm_loss = pd.read_csv('data/conv_knrm_lossResults.csv',index_col=0)
conv_knrm_loss = conv_knrm_loss.rename(columns={"0": "Loss"})

perf = pd.read_csv('data/performance.csv')

#read model results
mp_res = pd.read_csv('data/match_pyramid_firaRelevances.csv')
mp_res = mp_res.rename(columns={"Unnamed: 0": "query"})
knrm_res = pd.read_csv('data/knrm_firaRelevances.csv')
knrm_res = knrm_res.rename(columns={"Unnamed: 0": "query"})
conv_knrm_res = pd.read_csv('data/conv_knrm_firaRelevances.csv')
conv_knrm_res = conv_knrm_res.rename(columns={"Unnamed: 0": "query"})

#read fira_numsnippets_test_tuples to obtain id-text pairs
fira_tuples=pd.read_csv("data/fira_numsnippets_test_tuples.tsv",sep="\t",lineterminator='\n',header=None,names=['q','d','qT','dT'])

#create dataframe of query and doc ids and corresponding texts
queries = fira_tuples.drop_duplicates('q').loc[:,['q', 'qT']].set_index('q')
documents = fira_tuples.drop_duplicates('d').loc[:,['d', 'dT']].set_index('d')

# Merge validation results datasets
knrm = knrm_loss.merge(knrm_val, left_index=True, right_index=True, how='inner')
mp = mp_loss.merge(mp_val, left_index=True, right_index=True, how='inner')
conv_knrm = conv_knrm_loss.merge(conv_knrm_val, left_index=True, right_index=True, how='inner')
knrm['Model']="KNRM"
mp['Model']="MatchPyramid"
conv_knrm['Model']="CONV-KNRM"
res_df=pd.concat([knrm, mp,conv_knrm])

#list of possible measurements in the dropdown list
measurements = res_df.columns[0:len(res_df.columns)-1]

# list of possible models in the dropdown list
models = res_df['Model'].unique()

# set Treemap colorscale
colors=['#1a2543','#474049','#a27755','#cf925b','#FDAE61']


# make list of possible measurements for the measurements dropdown list
opt_eval = [{'label': i, 'value': i} for i in measurements]

# make list of possible models for the models dropdown list
opt_model = [{'label': i, 'value': i} for i in models]

# make list of queries for the models dropdown list
opt_query = [{'label': queries['qT'].loc[i], 'value': i} for i in queries.index]


# initialize dash app with some external styling
app = dash.Dash(__name__)

# build main title
def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.H6("IR model performance comparison")
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
                                        "Use the dropdown boxes to select models, measurements and queries. "
                                        "Use the graph's menu bar to select, zoom or save the plots."
                                    )
                                ),
                                html.Div(
                                    className="row",
                                    style={"marginBottom":"30px"},
                                    children=[
                                        html.Div(
                                            className="four columns",
                                            children=[
                                                build_graph_title("Model"),
                                                dcc.Dropdown(multi=False, clearable=False, value='KNRM',id = 'mod', options = opt_model)
                                            ]
                                        ),
                                        html.Div(
                                            className="eight columns",
                                            children=[
                                                build_graph_title("Evaluation metric"),
                                                dcc.Dropdown(multi=True, value=['MRR@10',"Loss"],id = 'eval', options = opt_eval)
                                            ]
                                        )
                                    ]
                                ),
                            ],
                        )
                    ],
                ),
                # indicator plot
                html.Div(
                    className="row",
                    id="top-row-graphs",
                    children=[                 
                        html.Div(
                            id="indicator-container",
                            className="four columns",
                            children=[
                                build_graph_title("Performance on MS MARCO"),
                                dcc.Graph(id='indicator')
                            ],
                        ),
                        # scatter plot
                        html.Div(
                            id="scatter-container",
                            className="eight columns",
                            children=[
                                html.Div(
                                    id="ternary-header",
                                    children=[
                                        build_graph_title(
                                            "Model evaluation on the MS MARCO validation set"
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
                # text retrieval
                html.Div(
                    id="text-container",
                    className="six columns",
                    children=[
                        build_graph_title(
                            "Select query from the FIRA dataset"
                        ),
                        dcc.Dropdown(multi=False, clearable=False, value=1112341,id = 'query', options = opt_query),
                        build_graph_title("Top 3 retrieved documents", style={'margin-top':'30px'}),
                        dcc.Loading(type="cube", style={'backgroundColor': '#f2f5fa', 'height': '450px', 'display': 'flex', 'alignItems': 'center'}, 
                        children=dcc.Markdown(id="markdown")),                                               
                    ],
                ),
                html.Div(
                    # text retrieval
                    id="word-container",
                    className="six columns",
                    children=[
                        build_graph_title("Common words in the retrieved documents"),
                        dcc.Loading(
                            type="cube", 
                            style={'backgroundColor': '#f2f5fa', 'height': '450px', 'display': 'flex', 'alignItems': 'center'},
                            children= dcc.Tabs(
                                id="tabs",
                                children=[
                                    dcc.Tab(label="Wordcloud",
                                        children=[
                                            dcc.Loading(
                                                id="loading-wordcloud",
                                                children=[
                                                    html.Img(id="wordcloud", style={'width':'100%','margin':'10px 0px'})
                                                ],
                                                type="cube",
                                            )
                                        
                                        ],
                                    ),
                                    dcc.Tab(label="Treemap",
                                        children=[
                                            dcc.Loading(
                                                id="loading-treemap",
                                                children=[dcc.Graph(id="treemap")],
                                                type="cube",
                                            )
                                        
                                        ],
                                    ),
                                ],
                            )
                        )
                    ],
                ),
            ],
        ),
    ]
)
# update plots 
@app.callback(
    [Output('scatter', 'figure'),Output('indicator','figure'),Output('markdown','children')],
    [Input('eval', 'value'), Input('mod', 'value'),Input('query', 'value')])
def update_figure(selected,mod,qID):

    # select measurement if no measurement is selected
    if len(selected) == 0:
        selected = ['MRR@10',"Recall@10"]

    # select model if no measurement is selected
    if len(mod) == 0:
        mod = ['KNRM']
    
    # filter dataset by model
    res_dff = res_df[res_df["Model"]==mod]

    # make scatter plot
    scatter_fig = plotly.subplots.make_subplots(
        subplot_titles=selected,
        rows=len(selected), 
        cols=1,
        shared_xaxes=True, 
        shared_yaxes=False,
        vertical_spacing=0.1,
    )

    trace = []
    scatter_fig.update_xaxes(row=len(selected), col=1,title_text="Iterations")  
    
    for i in range(len(selected)):
        #scatter_fig.update_yaxes(title_text=selected[i], row=i+1, col=1)
        trace.append(go.Scatter(
            x=res_dff.index, 
            y=res_dff[selected[i]], 
            name=selected[i], 
            mode='lines+markers', 
            connectgaps=True,
            marker={'size': 10, "opacity": 1, "line": {'width': 1}}
        ))
        scatter_fig.append_trace(trace[i], i+1, 1)
        
    scatter_fig['layout'].update(
        colorway=['#fdae61', '#800080', '#2c7bb6'], 
        margin=dict(l=50, r=20, t=30, b=40), 
        showlegend=False,
        paper_bgcolor='#f2f5fa'
    )


    #select data
    values=perf.loc[perf['Model'] == mod].iloc[:,0:len(perf.columns)-1].values.tolist()

    # make indicator plot

    fig_ind = go.Figure()

    fig_ind.add_trace(go.Indicator(
            value = values[0][0],
            title = {'text': "Train speed (it/s)",'font': {'size': 14},'align':'center'},
            gauge = {
            'axis': {'range': [0, 500]},
                'bar': {'color': "darkblue"},
                'bgcolor': "#fdae61",
            'bordercolor': "#1B2442",'borderwidth': 5,
            'steps': [
            {'range': [0, 250], 'color': 'cyan'},
            {'range': [250, 400], 'color': '#2C7BB6'}]},
            domain = {'row': 0, 'column': 0}))   

    fig_ind.add_trace(go.Indicator(
            value = values[0][1],
            title = {'text': "Validation speed (it/s)",'font': {'size': 14}},
            gauge = {
            'axis': {'range': [0, 500]},
                'bar': {'color': "darkblue"},
                'bgcolor': "#fdae61",
            'bordercolor': "#1B2442",'borderwidth': 5,
            'steps': [
            {'range': [0, 250], 'color': 'cyan'},
            {'range': [250, 400], 'color': '#2C7BB6'}]},
            domain = {'row': 1, 'column': 0}))    

    fig_ind.add_trace(go.Indicator(
            value = values[0][2],
            title = {'text': "Test speed (it/s)",'font': {'size': 14}},
            gauge = {
            'axis': {'range': [0, 500]},
                'bar': {'color': "darkblue"}, 
                'bgcolor': "#fdae61",
            'bordercolor': "#1B2442", 'borderwidth': 5,
            'steps': [
            {'range': [0, 250], 'color': 'cyan'},
            {'range': [250, 400], 'color': '#2C7BB6'}]},
            domain = {'row': 2, 'column': 0}))

    fig_ind.update_layout(
            paper_bgcolor = "#f2f5fa",font = dict(color="#192444",size=10),
            margin=dict(l=10, r=10, t=50, b=30),
            grid = {'rows': 3, 'columns': 1, 'pattern': "independent"},
            template = {'data' : {'indicator': [{
                'mode' : "number+gauge"
                }]
             }}
             )


    #retrive relevant document ids
    if (mod=="KNRM"):
        top_docs=knrm_res[knrm_res['query']==qID].sort_values(by=['relavance'],ascending=False).iloc[:3]['document']
    elif (mod=="MatchPyramid"):
        top_docs=mp_res[mp_res['query']==qID].sort_values(by=['relavance'],ascending=False).iloc[:3]['document']
    else: 
        top_docs=conv_knrm_res[conv_knrm_res['query']==qID].sort_values(by=['relavance'],ascending=False).iloc[:3]['document']

    doc_res=documents.loc[top_docs]
    docs = [doc_res['dT'].loc[i] for i in doc_res.index]

    #retrieve corresponding document text 
    doc1=re.sub('[^A-Za-z0-9 .,:/?!]+', '', str(docs[0]))
    doc2=re.sub('[^A-Za-z0-9 .,:/?!]+', '', str(docs[1]))
    doc3=re.sub('[^A-Za-z0-9 .,:/?!]+', '', str(docs[2]))
    md = ">"+doc1+"\n\n___\n\n>"+doc2+"\n\n___\n\n>"+doc3
    return  [scatter_fig, fig_ind,md]
    

# this callback controls the maximum selectable measurements
@app.callback(
    Output(component_id="eval", component_property="options"),
    [Input(component_id="eval", component_property="value")],
)
def update_dropdown_options(values):
    if len(values) == 3:
        return [option for option in opt_eval if option["value"] in values]
    else:
        return opt_eval


@app.callback(
    [
        Output("wordcloud", "src"),
        Output("treemap", "figure"),
    ],
    [
        Input('query', 'value'),
        Input('mod', 'value'),
    ],
)
def update_wordcloud_plot(qID,mod):
#    time.sleep(100)
    """ Callback to rerender wordcloud plot """
    #retrive relevant document ids
    if (mod=="KNRM"):
        top_docs=knrm_res[knrm_res['query']==qID].sort_values(by=['relavance'],ascending=False).iloc[:3]['document']
    elif (mod=="MatchPyramid"):
        top_docs=mp_res[mp_res['query']==qID].sort_values(by=['relavance'],ascending=False).iloc[:3]['document']
    else: 
        top_docs=conv_knrm_res[conv_knrm_res['query']==qID].sort_values(by=['relavance'],ascending=False).iloc[:3]['document']

    doc_res=documents.loc[top_docs]
    docs = [str(doc_res['dT'].loc[i]) for i in doc_res.index]
    docs[0]=re.sub('[^A-Za-z0-9 .,:/?!]+', '', str(docs[0]))
    docs[1]=re.sub('[^A-Za-z0-9 .,:/?!]+', '', str(docs[1]))
    docs[2]=re.sub('[^A-Za-z0-9 .,:/?!]+', '', str(docs[2]))

    text = ""
    wordcloud, treemap = plotly_wordcloud(text.join(docs))
    return (wordcloud, treemap)


def plotly_wordcloud(text):
 
    word_cloud = WordCloud(stopwords=set(STOPWORDS), max_words=100, max_font_size=90)
    word_cloud.generate(text)

    word_list = []
    freq_list = []

    for (word, freq), fontsize, position, orientation, color in word_cloud.layout_:
        word_list.append(word)
        freq_list.append(freq)

    word_list_top = word_list[:25]
    word_list_top.reverse()
    freq_list_top = freq_list[:25]
    freq_list_top.reverse()

    treemap_trace = go.Treemap(
        labels=word_list_top, parents=[""] * len(word_list_top), values=freq_list_top,marker_colorscale = colors
    )
    treemap_layout = go.Layout({"margin": dict(t=10, b=10, l=5, r=5, pad=4),'paper_bgcolor': "#f3f5f9"},)
    treemap_figure = {"data": [treemap_trace], "layout": treemap_layout}

    
    word_cloud = WordCloud(stopwords=set(STOPWORDS), max_words=100, max_font_size=90, background_color='#f3f5f9', width=1200, height=795, margin=20)
    word_cloud.generate(text)

    byte_io = BytesIO()
    word_cloud.to_image().save(byte_io, format='PNG')
    byte_io.seek(0)
    pic_hash = base64.b64encode(byte_io.read())

    wordcloud_figure_data = 'data:image/png;base64,'+str(pic_hash,'ascii')

    return wordcloud_figure_data, treemap_figure


# start the server if we are running the script by hand
if __name__ == '__main__':
    app.run_server(debug=True)

# pass the server instance for gunicorn in production
server = app.server
