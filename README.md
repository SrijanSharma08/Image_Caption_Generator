# Image_Caption_Generator
A deep learning based Image Caption Generator that extracts visual features using VGG16 and generates natural language captions using an LSTM model. Trained on the Flickr30k dataset.

# Image Caption Generator

This project generates natural language captions for images using VGG16 for feature extraction and an LSTM-based decoder network. The model is trained on the Flickr30k dataset to produce human-like descriptions for images.

---

## Features
- Image feature extraction using VGG16  
- Caption generation using LSTM and word embeddings  
- Flask backend for serving predictions  
- Simple HTML/CSS frontend  
- Training notebook included (ICG.ipynb)  
- Clean .gitignore to keep the repository lightweight  

---

## Tech Stack
- Python 3  
- TensorFlow / Keras  
- NumPy, Pandas, Pickle  
- Flask  
- HTML, CSS, JavaScript  

---

## Project Structure


Frontend/
│── main.py # Flask backend
│── index.html # Web interface
│── ICG.ipynb # Training and model development notebook
│── tokenizer.pkl # Tokenizer used during training and inference
│── max_length.pkl # Maximum caption length
│── results.pkl # Training history
│── features.pkl # Extracted VGG16 features
│── best_model.h5 # Trained model (ignored in repo)
│── static/ # CSS and JS assets
│── uploads/ # Uploaded images (ignored)
│── flickr30k_images/ # Dataset folder (ignored)
│── requirements.txt # Project dependencies
│── .gitignore # Ignored files and directories


---

## How to Run

### 1. Install Dependencies


pip install -r requirements.txt


### 2. Start the Flask Server


python main.py


### 3. Open the Application
Visit in your browser:


http://localhost:5000


Upload an image to generate a caption.

---

## Dataset
This project uses the Flickr30k dataset, which contains 30,000 images and five captions per image.  
The dataset is not included in this repository due to size.

---

## Model Overview
- Encoder: VGG16 pretrained on ImageNet  
- Decoder: LSTM with word embeddings  
- Loss: Categorical Cross-Entropy  
- Optimizer: Adam  
- Tokenization: Keras Tokenizer with padded sequences  

---

## Future Improvements
- Replace VGG16 with a better encoder such as InceptionV3 or EfficientNet  
- Implement beam search for higher-quality captions  
- Improve frontend UI or convert it to a Streamlit/React app  
- Deploy on HuggingFace Spaces, Render, or Docker  

---

## License
This project is free to use for educational and research purposes.
