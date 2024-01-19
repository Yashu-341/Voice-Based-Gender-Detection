import librosa
import numpy as np

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
    """

    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    X, sample_rate = librosa.core.load(file_name)

    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])

    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    return result

if __name__ == "__main__":

    from Dataset import create_model
    import PySimpleGUI as sg

    #get the audio file
    fname = sg.popup_get_file("choose Voice file", multiple_files=False, file_types=(("audio files", "*.wav*"),))
    if not fname:
        sg.popup("Cancel","no filename supplied")
        raise SystemExit("Cancelling ")

    file = fname
    print(type(fname))

    # construct the model
    model = create_model()
    # load the saved/trained weights
    model.load_weights(r"C:\Users\P Yasaswi\OneDrive\Desktop\NullClass\Project2\results\model.h5")

    features = extract_feature(file, mel=True).reshape(1, -1)
    # predict the gender!
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "male" if male_prob > female_prob else "female"
    # show the result!
    print("Result:", gender)
    print(f"Probabilities:     Male: {male_prob*100:.2f}%    Female: {female_prob*100:.2f}%")