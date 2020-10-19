from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField 
from flask_wtf.file import FileField, FileRequired, FileAllowed
from werkzeug.utils import secure_filename

class UploadForm(FlaskForm):
    radiografia = FileField('image', validators=[
        FileRequired('Empty file!'),
        FileAllowed(['jpg', 'png'], 'Images only!')
    ])
    submit = SubmitField('Upload')