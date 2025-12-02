import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import io, pickle, os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model

app = FastAPI()

# Middleware for CORS (cross-origin requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend on any domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and resources once on server start
model = load_model("new_best_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_length.pkl", "rb") as f:
    max_length = pickle.load(f)

# Mount static files (for serving index.html and uploaded images)
app.mount("/static", StaticFiles(directory="."), name="static")

# Serve the frontend
@app.get("/")
async def root():
    return FileResponse("index.html")

# Feature extraction function using VGG16
def encode_image(image_path):
    model_vgg = VGG16()
    model_vgg = Model(inputs=model_vgg.inputs, outputs=model_vgg.layers[-2].output)
    
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    
    feature = model_vgg.predict(image, verbose=0)
    return feature

# Caption generation function
def generate_caption_new(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()[2:-1]  # remove startseq and endseq
    return ' '.join(final)

# Upload and caption API
@app.post("/generate_caption")
async def generate_caption(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Ensure the static directory exists before saving
        os.makedirs("static", exist_ok=True)

        upload_path = "static/uploaded.jpg"
        image.save(upload_path)

        photo = encode_image(upload_path)
        caption = generate_caption_new(model, tokenizer, photo, max_length)

        return {
            "caption": caption,
            "image_url": "/static/uploaded.jpg"
        }

    except Exception as e:
        print("🔥 Error during caption generation:", str(e))
        return {
            "caption": "Error: " + str(e),
            "image_url": ""
        }
        
 