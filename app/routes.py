import os
from flask import render_template, redirect, url_for, request
from flask import current_app as app
from datetime import datetime
from app.forms import UploadForm, FlaskForm, secure_filename

@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')

@app.route("/bootstrap", methods=["GET"])
def bootstrap():
    return render_template('bootstrap.html')

@app.route("/hello")
@app.route("/hello/<name>")
def hello_there(name=None):
    return render_template("hello_there.html", name=name, date=datetime.now())

@app.route('/load_radio', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        f = form.radiografia.data
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_PHOTO_FOLDER'], filename))
        filerelpath=os.path.join('static/radio_uploads', filename)
        return render_template('load_radio.html', form=form, photo_url=filerelpath)

    return render_template('load_radio.html', form=form)