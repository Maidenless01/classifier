import gradio as gr
import tensorflow as tf
import numpy as np

# Load the model
model_path = r"c:\Users\ASUS\Desktop\classifier\cat_dog_model.keras"
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the prediction function
def classify_image(image):
    if model is None:
        return {"Error": "Model not loaded. Please train the model first."}
    
    if image is None:
        return {}

    # Resize image to match model input (160x160)
    image = image.resize((160, 160))
    # Convert image to numpy array
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    # Add batch dimension
    img_array = tf.expand_dims(img_array, 0)
    
    # Predict
    predictions = model.predict(img_array)
    
    # Get probability of being a dog
    prob_dog = float(predictions[0][0])
    prob_cat = 1.0 - prob_dog
    
    return {"Cat": prob_cat, "Dog": prob_dog}

# Create Gradio interface
ifce = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Cat vs Dog Image Classifier",
    description="Upload an image to identify whether it's a cat or a dog."
)

if __name__ == "__main__":
    ifce.launch()
