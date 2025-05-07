import os
import tensorflow as tf
import csv
from src.audio_processing import load_audio, extract_features, pad_features
from src.model_inference import load_model, load_metadata, predict_arousal_valence


# Ορισμός διαδρομής για το μοντέλο και τα metadata
model_path = 'models/deam-audioset-vggish-2.pb'
metadata_path = 'models/deam-audioset-vggish-2.json'

# Εκτυπώνουμε την πλήρη διαδρομή του μοντέλου για επαλήθευση
print(f"Model path: {os.path.abspath(model_path)}")
print(f"Metadata path: {os.path.abspath(metadata_path)}")

# Έλεγχος αν το μοντέλο υπάρχει
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
else:
    print(f"Model file found at {model_path}")

# Φόρτωμα του μοντέλου και των metadata
model = load_model(model_path)
metadata = load_metadata(metadata_path)

# Ορισμός διαδρομής για το αρχείο ήχου
audio_file = 'data/test1.mp3'

# Φόρτωμα του ήχου
audio = load_audio(audio_file)

# Εξαγωγή χαρακτηριστικών από τον ήχο
features = extract_features(audio)
features_padded = pad_features(features)
# Πρόβλεψη Arousal/Valence
arousal_valence = predict_arousal_valence(model, features_padded, metadata)

# Εκτύπωση των αποτελεσμάτων
print(f'The result is {arousal_valence[0]}, [Valence , Arousal]')

# Αποθήκευση αποτελεσμάτων σε αρχείο CSV
def save_results_to_csv(results, output_file='outputs/results.csv'):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Δημιουργεί το φάκελο 'outputs' αν δεν υπάρχει
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', '[Valence , Arousal]'])
        for result in results:
            writer.writerow(result)

# Στην main.py, μετά την πρόβλεψη, αποθήκευση των αποτελεσμάτων
results = [('test1.mp3', arousal_valence[0])]
save_results_to_csv(results)