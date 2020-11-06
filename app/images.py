import os
import numpy as np
import pandas as pd
import pickle
import copy
import cv2
import skimage
import skimage.io
import skimage.transform
import skimage.segmentation
import sklearn.metrics
from sklearn.linear_model import LinearRegression
from flask import current_app as app

def pred(img):
    try:
        if img is None:
            raise ValidationError('Imagen no ingresada')
        if app.config['PICKLE_FOLDER'] is None:
            raise ValidationError('Parametro sin configuracion')
        filename = 'model_lr_gs.pkl'
        filefolder = os.path.join(app.config['DATA_SOURCES_FOLDER'], filename)
        with open(filefolder, 'rb') as lr_model:
            modelo_logreg = pickle.load(lr_model)
        imgpred = cv2.imread(img, cv2.IMREAD_COLOR)
        imgpred = cv2.cvtColor(imgpred, cv2.COLOR_BGR2GRAY)
        imgpred = cv2.resize(imgpred, (100,100))
        imgpred = imgpred.flatten()
        y_pred = modelo_logreg.predict(imgpred.reshape(1,-1))[0]
        y_pred_proba = [ np.round(x,6) for x in modelo_logreg.predict_proba(imgpred.reshape(1,-1))[0]]
        if int(y_pred) == 2:
            y_pred = 'Neumonia Viral'
            copete = 'NEUMONIA VIRAL ATIPICA: La radiografía de tórax de su paciente presenta hallazgos compatibles con neumonía virales NO-COVID 19.'
            recomendacion = 'Se recomienda comunicarse con el médico derivante para comentarle los hallazgos. Reforzar con el paciente las medidas de cuidado personal (uso de barbijo, lavado de manos, distanciamiento), a fin de sobre-infectarse con COVID 19.'
        elif int(y_pred) == 0:
            y_pred = 'Covid-19'
            copete = 'COVID: La radiografía de tórax de su paciente presenta hallazgos compatibles con COVID-19 +.'
            recomendacion = 'Se recomienda comunicarse con el médico derivante para coordinar el manejo del paciente. Entregarle la placa con informe al paciente, indicarle aislamiento preventivo y comunicación con su médico.'
        else:
            y_pred = 'normal'
            copete = 'NORMAL: La radiografía de tórax de su paciente es normal.'
            recomendacion = 'No se encuentran hallazgos radiológicos de patología pulmonar infecciosa viral.'
        return y_pred, y_pred_proba, copete, recomendacion
    except ValidationError as err:
        return err

def limeImage(img):

    img_name = os.path.splitext(os.path.basename(img))[0] + '_lime' + os.path.splitext(os.path.basename(img))[1]

    img_load = skimage.io.imread(img)

    img_load = skimage.color.rgb2gray(img_load)
    img_load = skimage.transform.resize(img_load, (100,100))

    modelname = 'model_lr_gs.pkl'
    modelfolder = os.path.join(app.config['PICKLE_FOLDER'], modelname)

    with open(modelfolder, 'rb') as lr_model:
        modelo_logreg = pickle.load(lr_model)

    superpixels = skimage.segmentation.slic(img_load, n_segments=50, compactness=0.2, sigma=1, start_label=1)

    num_perturb = 150 # numero de perturbaciones
    trial_prob = 0.5 # probabilidad de cada intento
    times_test = 1 # cantidad de intentos

    num_superpixels = np.unique(superpixels).shape[0]
    perturbations = np.random.binomial(times_test, trial_prob, size=(num_perturb, num_superpixels))

    def perturb_image(img,perturbation,segments): 
        active_pixels = np.where(perturbation == 1)[0]
        mask = np.zeros(segments.shape)
        for active in active_pixels:
            mask[segments == active] = 1 
        perturbed_image = copy.deepcopy(img)
        perturbed_image = perturbed_image*mask
        return perturbed_image
    
    predictions = []
    for pert in perturbations:
        perturbed_img = perturb_image(img_load,pert,superpixels)
        rows,cols = perturbed_img.shape
        perturbed_img_1d = perturbed_img.reshape(rows*cols)
        pred = modelo_logreg.predict(perturbed_img_1d.reshape(1,-1))
        predictions.append(pred)

    predictions = np.array(predictions)
 
    original_image = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled 
    distances = sklearn.metrics.pairwise_distances(perturbations,original_image, metric='cosine').ravel()

    kernel_width = 0.25
    weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function

    simpler_model = LinearRegression()
    simpler_model.fit(X=perturbations, y=predictions, sample_weight=weights)
    coeff = simpler_model.coef_[0]

    num_top_features = 10
    top_features = np.argsort(coeff)[-num_top_features:] 

    mask = np.zeros(num_superpixels) 
    mask[top_features]= True #Activate top superpixels
    imglime = perturb_image(img_load,mask,superpixels)

    skimage.io.imsave(os.path.join(app.config['LIME_PHOTO_FOLDER'], img_name), imglime)

    return os.path.join('static/lime_photos', img_name)

def model_scores():
    df = pd.read_csv(os.path.join(app.config['DATA_SOURCES_FOLDER'], 'model_scores.csv'))
    return df

class ValidationError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
