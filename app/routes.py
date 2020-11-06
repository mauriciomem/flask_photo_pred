import os
from flask import render_template, redirect, url_for, request
from flask import current_app as app
from datetime import datetime
from app.images import pred, model_scores, limeImage
from app.forms import UploadForm, FlaskForm, secure_filename
from app.plots import pca_2d, pca_5d, model_barplot, pca_variance

@app.route("/", methods=["GET"])
def index():
    df = model_scores()
    idsplot, graph = model_barplot()
    return render_template(
        'index.html', 
        scores=df, 
        len=df.shape[0],
        idsplot=idsplot,
        graphjs=graph
    )

@app.route('/upload', methods=['GET', 'POST'])
def upload():

    form = UploadForm()
    if form.validate_on_submit():
        f = form.radiografia.data
        filename = secure_filename(f.filename)
        filefolder = os.path.join(app.config['UPLOAD_PHOTO_FOLDER'], filename)
        f.save(filefolder)
        filerelpath=os.path.join('static/radio_uploads', filename)
        predict, predict_proba, copete, recomendacion = pred(filefolder)
    
        lime_photo_url = limeImage(filefolder)

        return render_template('upload.html', 
                                form=form, 
                                photo_url=filerelpath, 
                                lime_photo_url=lime_photo_url, 
                                predict=predict, 
                                predict_proba=predict_proba,
                                copete=copete,
                                recomendacion=recomendacion
                                )

    return render_template('upload.html', form=form)

@app.route('/dimensions', methods=["GET"])
def dimensions():
    idsplot, graph = pca_2d()
    idsplot2, graph2 = pca_5d()
    idsplot3, graph3 = pca_variance()

    return render_template(
        'dimensions.html',
        idsplot=idsplot,
        graphjs=graph,
        idsplot2=idsplot2,
        graphjs2=graph2,
        idsplot3=idsplot3,
        graphjs3=graph3
    )