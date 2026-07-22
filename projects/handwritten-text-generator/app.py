import os
import pickle
import numpy as np
import tensorflow as tf
import streamlit as st

# Set up page configurations
st.set_page_config(page_title="Handwritten Text Generator", page_icon="✍️", layout="centered")

# Title and description
st.title("✍️ Handwritten Text Generator")
st.write("Generate character-by-character text based on patterns learned from handwriting datasets.")

# -----------------------------------------------------------------------------
# 1. Setup Paths & Load Resources
# -----------------------------------------------------------------------------
UI_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(UI_DIR)

model_path = os.path.join(BASE_DIR, "model.keras")
tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")

# Use a caching decorator so the model doesn't reload on every button click (saves CPU/Time)
@st.cache_resource
def load_resources():
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        return None, None
    
    loaded_model = tf.keras.models.load_model(model_path)
    char_map, index_map = pickle.load(open(tokenizer_path, "rb"))
    return loaded_model, (char_map, index_map)

model, mappings = load_resources()

# Error handling directly on the UI if files are missing
if model is None or mappings is None:
    st.error(f"⚠️ Files not found! Please ensure **model.keras** and **tokenizer.pkl** exist in the root folder:")
    st.code(BASE_DIR)
    st.stop()

char_to_index, index_to_char = mappings
SEQUENCE_LENGTH = 40  

# -----------------------------------------------------------------------------
# 2. Generation Logic
# -----------------------------------------------------------------------------
def generate_text(seed_text, length=200, temperature=0.5):
    generated = seed_text
    context = seed_text[-SEQUENCE_LENGTH:]
    while len(context) < SEQUENCE_LENGTH:
        context = " " + context

    for _ in range(length):
        x = np.array([[char_to_index.get(ch, 0) for ch in context]], dtype=np.int32)
        preds = model.predict(x, verbose=0)[0]
        
        preds = np.log(preds.astype("float64") + 1e-8) / temperature
        preds = np.exp(preds)
        preds /= preds.sum()
        
        next_index = np.random.choice(len(char_to_index), p=preds)
        next_char = index_to_char[next_index]
        
        generated += next_char
        context = context[1:] + next_char
    
    return generated

# -----------------------------------------------------------------------------
# 3. Streamlit UI Elements
# -----------------------------------------------------------------------------
st.divider()

# User Inputs
seed_input = st.text_input("Enter a Starting Seed Text:", value="Artificial Intelligence")
gen_length = st.slider("Number of Characters to Generate:", min_value=50, max_value=500, value=200, step=50)
temp_setting = st.slider("Creativity Temperature (Lower = Conservative, Higher = Wild):", min_value=0.2, max_value=1.2, value=0.5, step=0.1)

# Generate Action Button
if st.button("✨ Generate Text", type="primary"):
    if not seed_input.strip():
        st.warning("Please enter some seed text first!")
    else:
        with st.spinner("Analyzing patterns and writing..."):
            try:
                output_result = generate_text(seed_input, length=gen_length, temperature=temp_setting)
                
                st.subheader("📝 Generated Output:")
                st.info(output_result)
            except Exception as e:
                st.error(f"An error occurred during text generation: {e}")