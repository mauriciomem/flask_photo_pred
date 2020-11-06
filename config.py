import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'alta-password-re-segura'
    UPLOAD_PHOTO_FOLDER = os.path.join(basedir, 'app/static/radio_uploads')
    LIME_PHOTO_FOLDER = os.path.join(basedir, 'app/static/lime_photos')
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max-limit.
    DATA_SOURCES_FOLDER = os.path.join(basedir, 'data/sources') 
    PICKLE_FOLDER = os.path.join(basedir, 'data/models')