from pylsl import StreamInlet, resolve_byprop
import time
import torch
import numpy as np
import mne
import pandas as pd
from EEGNet import EEGNet
import keyboard


# Paramètres
Fs = 256            
n_channels = 4      
Wn = 1              
n_samples = int(Wn * Fs)  

# Charger le modèle PyTorch
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()  # Mettre le modèle en mode évaluation
    return model

# Fonction pour convertir un DataFrame en MNE
def convertDF2MNE(sub):
    info = mne.create_info(list(sub.columns), ch_types=['eeg'] * len(sub.columns), sfreq=256)
    info.set_montage('standard_1020')
    data = mne.io.RawArray(sub.T, info)
    data.set_eeg_reference()
    epochs = mne.make_fixed_length_epochs(data, duration=Wn, overlap=0.2 * Wn)
    return epochs.get_data()

# Chargement du modèle
model_path = './model/model.pth'  # Chemin vers ton modèle PyTorch
test_model = load_model(model_path)

# Fonction de prédiction
def predict(x_pred, model):
    x_pred = np.transpose(x_pred, (0, 2, 1))
    x_pred = x_pred[:, :, :, np.newaxis]
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32)  # Conversion en tensor PyTorch
    with torch.no_grad():  # Désactiver le calcul des gradients pendant la prédiction
        y_pred = model(x_pred_tensor)
    return y_pred.numpy()  # Convertir le tensor PyTorch en numpy array

# Fonction principale
def main():
    print("Recherche d'un flux EEG...")
    streams = resolve_byprop('type', 'EEG')
    inlet = StreamInlet(streams[0]) 
    samples = []

    # Collecter les premières données pour initialiser la prédiction
    for _ in range(n_samples):
        sample, _ = inlet.pull_sample()
        samples.append(sample[:-1])

    while True:
        time.sleep(1)  # Attendre un peu avant de récupérer le prochain échantillon
        sample, _ = inlet.pull_sample()
        samples.pop(0)  # Supprimer le plus ancien
        samples.append(sample[:-1])  # Ajouter le nouveau

        # Préparer les données pour la prédiction
        x_pred = np.array([samples])
        y_pred = predict(x_pred, test_model)

        # Prédiction et affichage du résultat
        if y_pred[0][0] > y_pred[0][1] and y_pred[0][0] > y_pred[0][2] and y_pred[0][0] > 0.85:
            print(f'Prédiction : Gauche (Confiance : {y_pred[0][0]:.2f})')
            keyboard.press('left')

        elif y_pred[0][1] > y_pred[0][0] and y_pred[0][1] > y_pred[0][2] and y_pred[0][1] > 0.95:
            print(f'Prédiction : Droite (Confiance : {y_pred[0][1]:.2f})')
            keyboard.press('right')
        else:
            print(f'Prédiction : Neutre (Confiance : {y_pred[0][2]:.2f})')
            keyboard.press('up')

# Exécution du programme
if __name__ == '__main__':
    main()
