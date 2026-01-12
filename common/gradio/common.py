import numpy as np
import tensorflow as tf
import warnings
from common.rag.common import generate_personality_summary

_model = None


def predict_handwriting(image):
    """
    Preprocess uploaded image exactly the way model expects
    """

    global _model

    if _model is None:
        _model = tf.keras.models.load_model(
            "signature_model_tfdata.keras",
            compile=False
        )

    if image is None:
        return "Please upload an image.", ""

    try:
        img = tf.keras.preprocessing.image.img_to_array(image)

        if img.shape[-1] == 4:
            img = img[..., :3]
        elif img.shape[-1] == 1:
            pass
        elif img.shape[-1] != 3:
            return "Unsupported image format (channels).", ""


        if img.shape[-1] == 3:
            img = tf.image.rgb_to_grayscale(img)

        IMG_SIZE = 224 
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = img / 255.0
        img = tf.image.grayscale_to_rgb(img)
        img = tf.expand_dims(img, axis=0)


        predictions = _model.predict(img, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx]) * 100

        CLASS_NAMES = [
            "Agreeableness",
            "Conscientiousness",
            "Extraversion",
            "Neuroticism",
            "Openness"
        ]

        trait = CLASS_NAMES[predicted_idx]

        result = f"**Predicted Personality Trait**\n{trait}\n\n**Confidence**: {confidence:.2f}%", trait
        return result

    except Exception as e:
        import traceback
        return f"Error during prediction:\n{str(e)}", ""
    
def full_analysis(image):
    if image is None:
        return "Please upload an image.", ""

    prediction_text, trait = predict_handwriting(image)
    summary = generate_personality_summary(trait) if trait else ""
    return prediction_text, summary