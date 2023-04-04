# Flask imports
import math

from flask import *
from flask_wtf import FlaskForm
from wtforms import FileField
from wtforms.validators import DataRequired

# Other imports
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import audioread
import os
# =============================================================================


# ======================================
# CONSTANTS
data_empty = {
    "mapping": [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock"
    ],
    "mfcc": [],
    "labels": []
}

sr = 22500
duration = 30  # duration in seconds
samples_per_track = sr * duration
num_segments = 6
hop_length = 512

num_samples_per_segment = int(samples_per_track / num_segments)
expected_num_mfcc_vectors_per_segment = math.ceil(
    num_samples_per_segment / hop_length)

# ======================================
most_common = lambda lst: max(set(lst), key=lst.count)
# =============================================================================
def load_model():
    model_version = ""
    while model_version not in ["6", "10", "11", "12"]:
        model_version = input("Select model version (6, 10 - 12):  ")
    model = tf.keras.models.load_model(f"./static/models/cnn_v{model_version}")
    return model


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
        file = FileField('File', validators=[DataRequired()])
        # FileRequired(),FileAllowed(images, 'Images only!')

    @app.route("/",methods=['GET', 'POST'])
    def index():
        form = UploadForm()
        if form.validate_on_submit():
            file = request.files['file']
            print(f"User selected file: {file.filename}")  # print a message
            file.save('./static/uploads/' + file.filename)  # save the file to the uploads folder
            print("File saved successfully")

            file_path = './static/uploads/' + file.filename
            audio_file = audioread.audio_open(file_path)

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
        return render_template('result.html', predicted_genre=predicted_genre,audio_file=audio_file, current_page=current_page)

    return app


def predict(model, x, data, y="Not given"):
    """ Predict the genre of a sample """
    x = x[np.newaxis, ...]
    prediction = model.predict(x)
    # print(prediction)
    # for i in range(len(prediction)):
    #     print("Genres: {}, Percentage: {}%".format(data["mapping"], prediction[i] * 100))

    # avoir l'index avec le plus d'occurences dans la prediction
    predicted_index = np.argmax(prediction, axis=1)
    music_genre = data["mapping"][predicted_index[0]]
    print("Expected : {}, Predicted genre : {} || index : {}".format(y, music_genre, predicted_index))
    return predicted_index, prediction


def file_mfcc(file_path, num_samples_per_segment, expected_num_mfcc_vectors_per_segment,
              dirpath=None,
              data=data_empty,
              hop_length=512, num_segments=5, n_mfcc=25, n_fft=2048,
              iterator=1,
              file_duration=30):
    """ Extracts mfcc from audio file and saves it into a json file along with genre labels. """
    signal, sample_rate = librosa.load(file_path, sr=sr)

    if file_duration > 90:
        # do not take the 60 first seconds
        signal = signal[len(signal) // 2: len(signal) // 2 + 30 * sample_rate]  # :(60 + 30 * sample_rate)

    # process all segments of audio file
    for segment in range(num_segments):
        start_sample = num_samples_per_segment * segment
        finish_sample = start_sample + num_samples_per_segment

        # store the mfcc for segment if it has the expected length
        mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                    sr=sample_rate,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length)

        mfcc = mfcc.T  # transpose the matrix
        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())
            data["labels"].append(iterator - 1)
            print("{}, segment:{}".format(file_path.split('\\')[-1], segment))


def predict_sample(model, data, inputs):

    stock = []
    predictions = []
    final_predictions = []
    final_predictions2 = []
    for i in range(len(inputs)):
        x = inputs[i]
        # X.reshape(132, 25, 1)
        # print(inputs.shape)

        predicted_index, prediction = predict(model, x, data)
        predictions.append(100 * np.array(prediction[0]))
        stock.append(predicted_index[0])

    predicted_genre = data["mapping"][most_common(stock)]
    final_predictions.append(np.mean(predictions, axis=0))
    print(" ============ ", end="\n\n")
    print("Predicted music genre is `{}`.".format(predicted_genre))
    print(stock)
    print(" ============ ", end="\n\n")
    return predicted_genre


# =============================================================================


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
