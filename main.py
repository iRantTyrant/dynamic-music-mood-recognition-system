import os
from src.audio_processing import load_audio, extract_features
from src.model_inference import load_model, predict_arousal_valence
import csv

# Ορισμός διαδρομής για το μοντέλο
model_path = 'models/vggish_model.pb'

# Φόρτωμα του μοντέλου
model = load_model(model_path)

# Ορισμός διαδρομής για το αρχείο ήχου
audio_file = 'data/test1.mp3'

# Φόρτωμα του ήχου
audio = load_audio(audio_file)

# Εξαγωγή χαρακτηριστικών
features = extract_features(audio)

# Πρόβλεψη Arousal/Valence
arousal_valence = predict_arousal_valence(model, features)

# Εκτύπωση των αποτελεσμάτων
print(f'Arousal: {arousal_valence[0]}, Valence: {arousal_valence[1]}')

# Αποθήκευση αποτελεσμάτων σε αρχείο CSV
def save_results_to_csv(results, output_file='outputs/results.csv'):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Arousal', 'Valence'])
        for result in results:
            writer.writerow(result)

# Στην main.py, μετά την πρόβλεψη
results = [('test1.mp3', arousal_valence[0], arousal_valence[1])]
save_results_to_csv(results)
