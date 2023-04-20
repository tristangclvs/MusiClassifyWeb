# Description: This file contains all the functions used in the myApp.py file
import math
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import audioread
import os
import soundfile as sf

# =============================================================================

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

outputs_folder = "uploads"

allowed_formats = ["mp3"]
# ======================================


most_common = lambda lst: max(set(lst), key=lst.count)


# =============================================================================

def load_model():
    model_version = ""
    while model_version not in ["6", "8", "10", "11", "12", "15"]:
        model_version = input("Select model version (6, 8, 10 - 12, 15):  ")
    # model_version = "12"
    model = tf.keras.models.load_model(f"./static/models/cnn_v{model_version}")
    return model


def predict(model, x, data, y="Not given"):
    """ Predict the genre of a sample """
    x = x[np.newaxis, ...]
    prediction = model.predict(x, verbose=0)
    print(prediction)
    for i in range(len(prediction)):
        print("Genres: {},\nPercentage: {}%".format(data["mapping"], prediction[i] * 100))

    # avoir l'index avec le plus d'occurences dans la prediction
    predicted_index = np.argmax(prediction, axis=1)
    music_genre = data["mapping"][predicted_index[0]]
    print("Expected : {}, Predicted genre : {} || index : {}".format(y, music_genre, predicted_index))
    return predicted_index, prediction


def file_mfcc(file, num_samples_per_segment, expected_num_mfcc_vectors_per_segment,
              dirpath=None,
              data=data_empty,
              hop_length=512, num_segments=5, n_mfcc=25, n_fft=2048,
              iterator=1,
              file_duration=30):
    """ Extracts mfcc from audio file and saves it into a json file along with genre labels. """

    # load audio file
    if dirpath is None:
        file_path = file
    else:
        file_path = os.path.join(dirpath, file)

    signal, sample_rate = librosa.load(file_path, sr=sr)

    if file_duration > 90:
        nb_segments = file_duration // 30
        for i in range(int(nb_segments)):
            # Set the start and end times for trimming (in seconds)
            start_time = 30 * i
            end_time = 30 * i + 30
            trimmed_audio = trim_audio(file_path, start_time, end_time)
            signal = trimmed_audio
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
    else:
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
        predicted_index, prediction = predict(model, x, data)
        predictions.append(100 * np.array(prediction[0]))
        stock.append(predicted_index[0])

    predicted_genre = data["mapping"][most_common(stock)]
    final_predictions.append(np.mean(predictions, axis=0))
    stock = []
    return predicted_genre


def trim_audio(abs_file_path: str, start_time: int, end_time: int, sr=22050) -> list:
    """ Trims an audio file and returns the trimmed audio """
    audio_path = abs_file_path if audio_to_wav(abs_file_path) is None else audio_to_wav(abs_file_path)
    signal, sample_rate = librosa.load(audio_path, sr=sr)
    # Calculate the start and end samples for trimming
    start_sample: int = int(start_time * sr)
    end_sample: int = int(end_time * sr)

    # Trim the audio file
    trimmed_audio = signal[start_sample:end_sample]

    return trimmed_audio


def audio_to_wav(file_path, outputs_folder=outputs_folder):
    """
    Converts audio file in .mp3 to .wav
    :param: file_path: absolute path to audio file
    :param: outputs_folder: folder where .wav is stored
    """
    try:
        # Get file format and assert its allowed
        file_format = file_path.split('.')[-1]
        assert file_format in allowed_formats
        # get the file name
        file_name = file_path.split('\\')[-1].replace("." + file_format, "")
        print("\n =========== ")
        print("Processing {0}".format(file_name))
        print(" =========== \n")
        audio, sr = sf.read(file_path)
        output_file = "{0}.wav".format(file_name)
        # sf.write("outputs/{0}".format(output_file), audio, sr)
        sf.write("{0}/{1}".format(outputs_folder, output_file), audio, sr)
        print("Conversion done ! ")
        print(output_file)

    except AssertionError:
        print('DUT DUT ERROR, THIS IS NOT A MP3 FILE')
        return None

    return os.path.join(os.getcwd(), "outputs", output_file)
