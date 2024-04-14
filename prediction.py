import numpy as np
import tensorflow as tf
from ai.dataset import get_description
import json


model = tf.keras.models.load_model('ai/trained_model.keras')
tokenizer = tf.keras.preprocessing.text.Tokenizer()

file_path = 'output/texts/extracted_text.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

max_length = max(len(seq) for seq in sequences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post')

predictions = model.predict(padded_sequences)

threshold = np.mean(predictions)  # Adjust threshold dynamically based on the mean prediction
classified_predictions = [get_description(1) if prediction >= threshold else get_description(0) for prediction in predictions.flatten()]

predictions_by_value = {}

for line, prediction in zip(lines, classified_predictions):
    if prediction not in predictions_by_value:
        predictions_by_value[prediction] = []

    predictions_by_value[prediction].append(line.strip())

json_data = json.dumps(predictions_by_value, indent=4)

with open('output/result/result.json', 'w') as f:
    f.truncate(0)
    f.write(json_data)

print(f"Predictions saved to: output/result/result.json")