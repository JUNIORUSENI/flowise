import pandas as pd
import numpy as np

# Génération de données aléatoires
np.random.seed(42)
nombre_echantillons = 100
donnees = {
    'Magnitude': np.random.uniform(2, 8, nombre_echantillons),  # Magnitude du séisme
    'Profondeur': np.random.uniform(1, 100, nombre_echantillons),  # Profondeur du séisme (km)
    'Latitude': np.random.uniform(-90, 90, nombre_echantillons),  # Latitude
    'Longitude': np.random.uniform(-180, 180, nombre_echantillons),  # Longitude
    'Séisme': np.random.choice(['Probable', 'Improbable'], nombre_echantillons)  # Cible
}

# Création du DataFrame
df = pd.DataFrame(donnees)

# Sauvegarde en fichier Excel
df.to_excel('donnees_seismes.xlsx', index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Chargement du dataset
df = pd.read_excel('donnees_seismes.xlsx')

# Séparation des caractéristiques et de la cible
caracteristiques = df.drop('Séisme', axis=1)  # Caractéristiques
cible = df['Séisme']  # Cible (probabilité de séisme)

# Encodage des labels de la cible en valeurs numériques
encodeur_labels = LabelEncoder()
cible_encodee = encodeur_labels.fit_transform(cible)

# Division des données en ensembles d'entraînement et de test
caracteristiques_entrainement, caracteristiques_test, cible_entrainement, cible_test = train_test_split(
    caracteristiques, cible_encodee, test_size=0.2, random_state=42
)

# Création et entraînement du modèle de classification (forêt aléatoire)
modele = RandomForestClassifier(random_state=42)
modele.fit(caracteristiques_entrainement, cible_entrainement)

# Prédiction sur l'ensemble de test
cible_predite = modele.predict(caracteristiques_test)

# Évaluation du modèle
precision = accuracy_score(cible_test, cible_predite)
rapport_classification = classification_report(cible_test, cible_predite, target_names=encodeur_labels.classes_)

print(f'Précision : {precision}')
print(f'Rapport de Classification :\n{rapport_classification}')


def predire_seisme(magnitude, profondeur, latitude, longitude):
    """
    Fonction pour prédire la probabilité d'un séisme en fonction des données sismiques.
    
    Paramètres :
    - magnitude : Magnitude du séisme
    - profondeur : Profondeur du séisme (km)
    - latitude : Latitude
    - longitude : Longitude
    
    Retour :
    - La probabilité de séisme (Probable ou Improbable)
    """
    # Création d'un DataFrame avec les valeurs d'entrée
    nouvel_echantillon = pd.DataFrame({
        'Magnitude': [magnitude],
        'Profondeur': [profondeur],
        'Latitude': [latitude],
        'Longitude': [longitude]
    })
    
    # Prédiction
    seisme_predite_encode = modele.predict(nouvel_echantillon)
    seisme_predite = encodeur_labels.inverse_transform(seisme_predite_encode)
    return seisme_predite[0]

# Exemple d'utilisation
seisme_predite = predire_seisme(6.5, 50, 35, -120)
print(f'Probabilité de séisme prédite : {seisme_predite}')












import pandas as pd
import numpy as np

# Génération de données aléatoires
np.random.seed(42)
nombre_echantillons = 100
donnees = {
    'Résistivité': np.random.uniform(1, 1000, nombre_echantillons),  # Résistivité (Ohm.m)
    'Densité': np.random.uniform(1, 3, nombre_echantillons),  # Densité (g/cm³)
    'Vitesse_Ondes': np.random.uniform(1, 8, nombre_echantillons),  # Vitesse des ondes sismiques (km/s)
    'Type_Roche': np.random.choice(['Igneux', 'Sédimentaire', 'Métamorphique'], nombre_echantillons)  # Cible
}

# Création du DataFrame
df = pd.DataFrame(donnees)

# Sauvegarde en fichier Excel
df.to_excel('donnees_roches_geophysiques.xlsx', index=False)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Chargement du dataset
df = pd.read_excel('donnees_roches_geophysiques.xlsx')

# Séparation des caractéristiques et de la cible
caracteristiques = df.drop('Type_Roche', axis=1)  # Caractéristiques
cible = df['Type_Roche']  # Cible (type de roche)

# Encodage des labels de la cible en valeurs numériques
encodeur_labels = LabelEncoder()
cible_encodee = encodeur_labels.fit_transform(cible)

# Division des données en ensembles d'entraînement et de test
caracteristiques_entrainement, caracteristiques_test, cible_entrainement, cible_test = train_test_split(
    caracteristiques, cible_encodee, test_size=0.2, random_state=42
)

# Création et entraînement du modèle de classification (forêt aléatoire)
modele = RandomForestClassifier(random_state=42)
modele.fit(caracteristiques_entrainement, cible_entrainement)

# Prédiction sur l'ensemble de test
cible_predite = modele.predict(caracteristiques_test)

# Évaluation du modèle
precision = accuracy_score(cible_test, cible_predite)
rapport_classification = classification_report(cible_test, cible_predite, target_names=encodeur_labels.classes_)

print(f'Précision : {precision}')
print(f'Rapport de Classification :\n{rapport_classification}')


def predire_type_roche(resistivite, densite, vitesse_ondes):
    """
    Fonction pour prédire le type de roche en fonction des données géophysiques.
    
    Paramètres :
    - resistivite : Résistivité (Ohm.m)
    - densite : Densité (g/cm³)
    - vitesse_ondes : Vitesse des ondes sismiques (km/s)
    
    Retour :
    - Le type de roche prédit (Igneux, Sédimentaire, Métamorphique)
    """
    # Création d'un DataFrame avec les valeurs d'entrée
    nouvel_echantillon = pd.DataFrame({
        'Résistivité': [resistivite],
        'Densité': [densite],
        'Vitesse_Ondes': [vitesse_ondes]
    })
    
    # Prédiction
    type_roche_predite_encode = modele.predict(nouvel_echantillon)
    type_roche_predite = encodeur_labels.inverse_transform(type_roche_predite_encode)
    return type_roche_predite[0]

# Exemple d'utilisation
type_roche_predite = predire_type_roche(500, 2.5, 5.5)
print(f'Type de roche prédit : {type_roche_predite}')












import pandas as pd
import numpy as np

# Génération de données aléatoires
np.random.seed(42)
nombre_echantillons = 100
donnees = {
    'Conductivité': np.random.uniform(0.1, 10, nombre_echantillons),  # Conductivité (S/m)
    'Susceptibilité_Magnétique': np.random.uniform(0, 1, nombre_echantillons),  # Susceptibilité magnétique
    'Densité': np.random.uniform(1, 5, nombre_echantillons),  # Densité (g/cm³)
    'Ressource': np.random.choice(['Présente', 'Absente'], nombre_echantillons)  # Cible
}

# Création du DataFrame
df = pd.DataFrame(donnees)

# Sauvegarde en fichier Excel
df.to_excel('donnees_ressources_minerales.xlsx', index=False)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Chargement du dataset
df = pd.read_excel('donnees_ressources_minerales.xlsx')

# Séparation des caractéristiques et de la cible
caracteristiques = df.drop('Ressource', axis=1)  # Caractéristiques
cible = df['Ressource']  # Cible (présence de ressources)

# Encodage des labels de la cible en valeurs numériques
encodeur_labels = LabelEncoder()
cible_encodee = encodeur_labels.fit_transform(cible)

# Division des données en ensembles d'entraînement et de test
caracteristiques_entrainement, caracteristiques_test, cible_entrainement, cible_test = train_test_split(
    caracteristiques, cible_encodee, test_size=0.2, random_state=42
)

# Création et entraînement du modèle de classification (forêt aléatoire)
modele = RandomForestClassifier(random_state=42)
modele.fit(caracteristiques_entrainement, cible_entrainement)

# Prédiction sur l'ensemble de test
cible_predite = modele.predict(caracteristiques_test)

# Évaluation du modèle
precision = accuracy_score(cible_test, cible_predite)
rapport_classification = classification_report(cible_test, cible_predite, target_names=encodeur_labels.classes_)

print(f'Précision : {precision}')
print(f'Rapport de Classification :\n{rapport_classification}')



def predire_ressource(conductivite, susceptibilite_magnetique, densite):
    """
    Fonction pour prédire la présence de ressources minérales en fonction des données géophysiques.
    
    Paramètres :
    - conductivite : Conductivité (S/m)
    - susceptibilite_magnetique : Susceptibilité magnétique
    - densite : Densité (g/cm³)
    
    Retour :
    - La présence de ressources minérales (Présente ou Absente)
    """
    # Création d'un DataFrame avec les valeurs d'entrée
    nouvel_echantillon = pd.DataFrame({
        'Conductivité': [conductivite],
        'Susceptibilité_Magnétique': [susceptibilite_magnetique],
        'Densité': [densite]
    })
    
    # Prédiction
    ressource_predite_encode = modele.predict(nouvel_echantillon)
    ressource_predite = encodeur_labels.inverse_transform(ressource_predite_encode)
    return ressource_predite[0]

# Exemple d'utilisation
ressource_predite = predire_ressource(5, 0.5, 3)
print(f'Présence de ressources minérales prédite : {ressource_predite}')









