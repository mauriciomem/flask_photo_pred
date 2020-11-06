import json
import os
import pandas as pd
import numpy as np
from flask import current_app as app
import plotly
import plotly.graph_objs as go
import plotly.express as px
from sklearn.decomposition import PCA

def pca_2d():

    df_imagenes = pd.read_csv(os.path.join(app.config['DATA_SOURCES_FOLDER'], 'dataframe_imagenes.zip'), compression='zip')

    df_imagenes.loc[df_imagenes.target == 0, 'target'] = 'Covid-19'
    df_imagenes.loc[df_imagenes.target == 1, 'target'] = 'Normal'
    df_imagenes.loc[df_imagenes.target == 2, 'target'] = 'Neumonia'

    pca2 = PCA(n_components=2)
    components2 = pca2.fit_transform(df_imagenes.drop(['target'], axis=1))

    total_var = pca2.explained_variance_ratio_.sum() * 100

    fig2 = px.scatter(components2, x=0, y=1, color=df_imagenes.target, labels={'0': 'PC 1', '1': 'PC 2'})

    fig2.update_layout(title=f'Varianza total explicada: {total_var:.2f}%',
                    width=650,
                    height=600
                    )

    ids = 'PCA TOP 2 Principal Components Matrix'

    graphJSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    return ids, graphJSON

def pca_5d():

    df_imagenes = pd.read_csv(os.path.join(app.config['DATA_SOURCES_FOLDER'], 'dataframe_imagenes.zip'), compression='zip')

    df_imagenes.loc[df_imagenes.target == 0, 'target'] = 'Covid-19'
    df_imagenes.loc[df_imagenes.target == 1, 'target'] = 'Normal'
    df_imagenes.loc[df_imagenes.target == 2, 'target'] = 'Neumonia'

    pca = PCA(n_components=5)
    components = pca.fit_transform(df_imagenes.drop(['target'], axis=1))
    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(components.shape[1]),
        color=df_imagenes.target,
    )

    fig.update_layout(title='Top 5 PC',
                    width=800,
                    height=800
                    )

    fig.update_traces(diagonal_visible=False)

    ids='PCA TOP 5 Principal Components'

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return ids, graphJSON

def pca_variance():

    df_imagenes = pd.read_csv(os.path.join(app.config['DATA_SOURCES_FOLDER'], 'dataframe_imagenes.zip'), compression='zip')

    df_imagenes.loc[df_imagenes.target == 0, 'target'] = 'Covid-19'
    df_imagenes.loc[df_imagenes.target == 1, 'target'] = 'Normal'
    df_imagenes.loc[df_imagenes.target == 2, 'target'] = 'Neumonia'

    pca = PCA()
    pca.fit(df_imagenes.drop(['target'], axis=1))

    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

    fig = px.area(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        labels={"x": "nro componentes principales", "y": "% varianza explicada"}
    )

    ids='Varianza Explicada'

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return ids, graphJSON

def model_barplot():

    df = pd.read_csv(os.path.join(app.config['DATA_SOURCES_FOLDER'], 'model_scores_list.csv'))

    fig = px.bar(df, x='metrics', y='score', color='model', barmode="group")
   
    ids='Classification Models Grouped Barplots'

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return ids, graphJSON
