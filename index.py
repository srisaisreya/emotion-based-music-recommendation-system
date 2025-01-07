
from flask import Flask, flash, redirect, render_template, request, url_for, Response
import operator
import cv2
import random
import os
from keras.models import load_model
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Define quotes
aquote = ['where words fail music speaks', 'lose yourself in the music', 'music on world off', 'music is the best therapist']
hquote = ["Music is life, that's why our hearts have beats", 'let see if we can get your day even more fun', 'Come, let us sing for joy', 'Music causes us to think eloquently']
nquote = ['Lose your dream, you lose your mind', 'I think there is a song out there to describe just about any situation', 'When you wake up with a song stuck in your head, it means an angel sang you to sleep', 'those who wish to sing always find a song']
squote = ['Our sweetest songs are those of saddest thought', "If everyone started off the day singing, just think how happy they'd be", 'Life is a wheel of fortune and it’s your turn to spin it', 'Life is a song sing it']

a = None
data = {'english': 1, 'telugu': 0, 'hindi': 0}

# Load pre-trained emotion detection model
emotion_model = load_model('emotion_model.h5')  # Replace with path to your pre-trained model
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        cv2.imwrite('img.jpg', frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route("/test", methods=['GET', 'POST'])
def test():
    select = request.form.get('comp_select')
    global a, data
    a = {'english': 0, 'telugu': 0, 'hindi': 0}
    for i in a:
        a[i] = 0
    a[str(select)] = 1
    a = dict(sorted(a.items(), key=operator.itemgetter(1), reverse=True))
    data = a
    return redirect(url_for('index'))

@app.route('/cam')
def cam():
    return render_template('cam.html', data=data)

@app.route('/songs')
def emo():
    s = []
    q = None
    # EMOTION DETECTION:
    img = cv2.imread("img.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    emo = 'happy'  # Default emotion for testing

    # Assume we detect only the first face in the image
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        face = cv2.resize(face, (48, 48))  # Resize to the size that the model expects
        face = np.stack([face] * 3, axis=-1)  # Convert grayscale to 3 channels (RGB)
        face = face / 255.0  # Normalize the image
        face = np.reshape(face, (1, 48, 48, 3))  # Add batch dimension

        # Predict emotion
        emotion_prediction = emotion_model.predict(face)
        max_index = np.argmax(emotion_prediction[0])
        emo = emotion_labels[max_index]  # Get predicted emotion

    lan = max(data, key=data.get)

    # Select songs based on language and emotion
    if lan == 'hindi':
        if emo == 'angry':
            s = random.sample(os.listdir('./static/hindi/angry'), 5)
            q = random.sample(aquote, 1)
        elif emo == 'neutral':
            s = random.sample(os.listdir('./static/hindi/neutral'), 5)
            q = random.sample(nquote, 1)
        elif emo == 'happy':
            s = random.sample(os.listdir('./static/hindi/happy'), 5)
            q = random.sample(hquote, 1)
        elif emo == 'sad':
            s = random.sample(os.listdir('./static/hindi/sad'), 5)
            q = random.sample(squote, 1)

    elif lan == 'english':
        if emo == 'angry':
            s = random.sample(os.listdir('./static/english/angry'), 5)
            q = random.sample(aquote, 1)
        elif emo == 'neutral':
            s = random.sample(os.listdir('./static/english/neutral'), 5)
            q = random.sample(nquote, 1)
        elif emo == 'happy':
            s = random.sample(os.listdir('./static/english/happy'), 5)
            q = random.sample(hquote, 1)
        elif emo == 'sad':
            s = random.sample(os.listdir('./static/english/sad'), 5)
            q = random.sample(squote, 1)

    elif lan == 'telugu':
        if emo == 'angry':
            s = random.sample(os.listdir('./static/telugu/angry'), 5)
            q = random.sample(aquote, 1)
        elif emo == 'neutral':
            s = random.sample(os.listdir('./static/telugu/neutral'), 5)
            q = random.sample(nquote, 1)
        elif emo == 'happy':
            s = random.sample(os.listdir('./static/telugu/happy'), 5)
            q = random.sample(hquote, 1)
        elif emo == 'sad':
            s = random.sample(os.listdir('./static/telugu/sad'), 5)
            q = random.sample(squote, 1)

    return render_template('songs.html', emotion=emo, songs=s, quote=q[0], lan=lan, data=data)

@app.route('/')
def index():
    if a != None:
        return render_template('index.html', data=a)
    else:
        return render_template('index.html', data=data)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
