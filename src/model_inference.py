import tensorflow as tf
import json
import numpy as np

# Φόρτωση frozen graph
def load_model(model_path):
    # Χρησιμοποιούμε το tf.compat.v1 για backward compatibility
    tf.compat.v1.disable_eager_execution()  # Απενεργοποιούμε την eager execution για την φόρτωση του frozen model

    # Δημιουργία ενός empty graph
    graph = tf.Graph()

    with graph.as_default():
        # Φόρτωση του frozen model
        with tf.io.gfile.GFile(model_path, 'rb') as f:  # Χρησιμοποιούμε το tf.io.gfile.GFile αντί για το παλιό tf.gfile.GFile
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')  # Φορτώνουμε το graph στο υπάρχον graph
    
    return graph  # Επιστρέφουμε το graph για χρήση στη συνέχεια

# Φόρτωση των metadata
def load_metadata(metadata_path):
    # Διαβάζουμε το αρχείο json που περιέχει τα metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

# Συνάρτηση για την πρόβλεψη Arousal/Valence
def predict_arousal_valence(model, features, metadata):
    # Χρησιμοποιούμε το γράφημα που έχει φορτωθεί με το μοντέλο
    input_tensor_name = metadata['schema']['inputs'][0]['name']
    output_tensor_name = metadata['schema']['outputs'][0]['name']
    
    # Το γράφημα πρέπει να έχει ήδη το input_tensor και output_tensor
    with model.as_default():  # Χρησιμοποιούμε το γράφημα με την εντολή as_default()
        # Ανακτούμε τα tensors από το γράφημα
        input_tensor = model.get_tensor_by_name(input_tensor_name + ':0')
        output_tensor = model.get_tensor_by_name(output_tensor_name + ':0')

        # Δημιουργούμε μια συνεδρία (session) για να κάνουμε την πρόβλεψη
        with tf.compat.v1.Session(graph=model) as sess:
            # Εκτελούμε το graph και κάνουμε την πρόβλεψη
            prediction = sess.run(output_tensor, feed_dict={input_tensor: features})
            
            # Επιστρέφουμε την πρόβλεψη
            return prediction
