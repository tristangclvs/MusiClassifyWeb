# Flask imports
import math

from flask import *
from flask_wtf import FlaskForm
from wtforms import FileField
from wtforms.validators import DataRequired
from flask_wtf.file import FileField, FileAllowed, FileRequired

# Other imports
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import audioread
import os

from static.utils import *


# =============================================================================

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'testflaskapp'
    model = load_model()
    print('Model loaded. ')
    print(model.summary())

    # Clean the uploads folder on startup
    for file in os.listdir('./static/uploads'):
        os.remove('./static/uploads/' + file)

    class UploadForm(FlaskForm):
        file = FileField('File', validators=[
            DataRequired(),
            FileRequired(),
            FileAllowed(['mp4', 'wav', 'mp3', 'flac'], 'Audio files only!')
        ])

        # FileRequired(),FileAllowed(images, 'Images only!')

    @app.route("/", methods=['GET', 'POST'])
    def index():
        form = UploadForm()
        if form.validate_on_submit():
            file = request.files['file']
            print(f"User selected file: {file.filename}")  # print a message
            file.save('./static/uploads/' + file.filename)  # save the file to the uploads folder
            print("File saved successfully")

            file_path = './static/uploads/' + file.filename
            audio_file = audioread.audio_open(file_path)
            print("File duration : {} seconds".format(audio_file.duration))

            file_mfcc(file_path, num_samples_per_segment=num_samples_per_segment,
                      expected_num_mfcc_vectors_per_segment=expected_num_mfcc_vectors_per_segment,
                      dirpath=None,
                      data=data_empty,
                      hop_length=512, num_segments=num_segments, n_mfcc=25, n_fft=2048,
                      iterator=1,
                      file_duration=audio_file.duration)
            inputs = np.array(data_empty["mfcc"])
            targets = np.array(data_empty["labels"])

            predicted_genre = predict_sample(model, data_empty, inputs)
            result_html = result(predicted_genre, file_path)
            # return redirect(url_for('result', predicted_genre=predicted_genre))
            return result_html
        return render_template('home.html', form=form, current_page="home")
        # current_page = "home"
        # return render_template('home.html', current_page=current_page)

    @app.route("/result")
    def result(predicted_genre, audio_file):
        current_page = "result"
        print(audio_file)
        return render_template('result.html', predicted_genre=predicted_genre, audio_file=audio_file,
                               current_page=current_page)

    return app


# =============================================================================


if __name__ == '__main__':
    app = create_app()
    app.run()
