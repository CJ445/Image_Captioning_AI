# testing
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, add
import numpy as np
import pickle

# Step 1: Extract Features from Images
def extract_features(image_path, model):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    features = model.predict(image, verbose=0)
    return features

# Load pre-trained model
image_model = VGG16()
image_model = Model(inputs=image_model.inputs, outputs=image_model.layers[-2].output)

# Step 2: Prepare Caption Data (loading a preprocessed dataset for simplicity)
with open('captions.pkl', 'rb') as f:
    captions_data = pickle.load(f)

tokenizer = captions_data['tokenizer']
max_length = captions_data['max_length']
vocab_size = len(tokenizer.word_index) + 1

# Step 3: Build the Caption Generator Model
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,))
    fe1 = Dense(256, activation='relu')(inputs1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = LSTM(256)(se1)
    decoder1 = add([fe1, se2])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

caption_model = define_model(vocab_size, max_length)
caption_model.compile(loss='categorical_crossentropy', optimizer='adam')

# Load model weights
caption_model.load_weights('caption_model.h5')

# Step 4: Generate Captions for New Images
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

# Generate caption for a new image
image_path = 'new_image.jpg'
photo = extract_features(image_path, image_model)
caption = generate_caption(caption_model, tokenizer, photo, max_length)
print('Generated Caption:', caption)
        