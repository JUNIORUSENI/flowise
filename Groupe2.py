import pandas as pd
import numpy as np

# Génération de données aléatoires
np.random.seed(42)
nombre_echantillons = 100
donnees = {
    'SiO2': np.random.uniform(40, 80, nombre_echantillons),  # Dioxyde de silicium
    'Al2O3': np.random.uniform(10, 20, nombre_echantillons),  # Alumine
    'MgO': np.random.uniform(1, 10, nombre_echantillons),  # Oxyde de magnésium
    'CaO': np.random.uniform(1, 15, nombre_echantillons),  # Oxyde de calcium
    'Na2O': np.random.uniform(1, 5, nombre_echantillons),  # Oxyde de sodium
    'K2O': np.random.uniform(1, 5, nombre_echantillons),  # Oxyde de potassium
    'Fe2O3': np.random.uniform(1, 20, nombre_echantillons)  # Oxyde de fer (cible)
}

# Création du DataFrame
df = pd.DataFrame(donnees)

# Sauvegarde en fichier Excel
df.to_excel('donnees_geochimie.xlsx', index=False)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Chargement du dataset
df = pd.read_excel('donnees_geochimie.xlsx')

# Séparation des caractéristiques et de la cible
caracteristiques = df.drop('Fe2O3', axis=1)  # Caractéristiques
cible = df['Fe2O3']  # Cible (concentration de Fe2O3)

# Division des données en ensembles d'entraînement et de test
caracteristiques_entrainement, caracteristiques_test, cible_entrainement, cible_test = train_test_split(
    caracteristiques, cible, test_size=0.2, random_state=42
)

# Création et entraînement du modèle de régression linéaire
modele = LinearRegression()
modele.fit(caracteristiques_entrainement, cible_entrainement)

# Prédiction sur l'ensemble de test
cible_predite = modele.predict(caracteristiques_test)

# Évaluation du modèle
erreur_quadratique_moyenne = mean_squared_error(cible_test, cible_predite)
score_r2 = r2_score(cible_test, cible_predite)

print(f'Erreur Quadratique Moyenne (MSE): {erreur_quadratique_moyenne}')
print(f'Score R2: {score_r2}')


def predire_concentration_fe2o3(SiO2, Al2O3, MgO, CaO, Na2O, K2O):
    """
    Fonction pour prédire la concentration de Fe2O3 en fonction des caractéristiques chimiques.
    
    Paramètres :
    - SiO2 : Concentration de dioxyde de silicium
    - Al2O3 : Concentration d'alumine
    - MgO : Concentration d'oxyde de magnésium
    - CaO : Concentration d'oxyde de calcium
    - Na2O : Concentration d'oxyde de sodium
    - K2O : Concentration d'oxyde de potassium
    
    Retour :
    - La concentration prédite de Fe2O3
    """
    # Création d'un DataFrame avec les valeurs d'entrée
    nouvel_echantillon = pd.DataFrame({
        'SiO2': [SiO2],
        'Al2O3': [Al2O3],
        'MgO': [MgO],
        'CaO': [CaO],
        'Na2O': [Na2O],
        'K2O': [K2O]
    })
    
    # Prédiction
    concentration_predite = modele.predict(nouvel_echantillon)
    return concentration_predite[0]

# Exemple d'utilisation
concentration_predite = predire_concentration_fe2o3(50, 15, 5, 10, 3, 2)
print(f'Concentration prédite de Fe2O3 : {concentration_predite}')












import pandas as pd
import numpy as np

# Génération de données aléatoires
np.random.seed(42)
nombre_echantillons = 100
donnees = {
    'SiO2': np.random.uniform(40, 80, nombre_echantillons),  # Dioxyde de silicium
    'Al2O3': np.random.uniform(10, 20, nombre_echantillons),  # Alumine
    'Fe2O3': np.random.uniform(1, 20, nombre_echantillons),  # Oxyde de fer
    'MgO': np.random.uniform(1, 10, nombre_echantillons),  # Oxyde de magnésium
    'CaO': np.random.uniform(1, 15, nombre_echantillons),  # Oxyde de calcium
    'Na2O': np.random.uniform(1, 5, nombre_echantillons),  # Oxyde de sodium
    'K2O': np.random.uniform(1, 5, nombre_echantillons),  # Oxyde de potassium
    'Type': np.random.choice(['Igneux', 'Sédimentaire', 'Métamorphique'], nombre_echantillons)  # Cible
}

# Création du DataFrame
df = pd.DataFrame(donnees)

# Sauvegarde en fichier Excel
df.to_excel('donnees_types_roches.xlsx', index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Chargement du dataset
df = pd.read_excel('donnees_types_roches.xlsx')

# Séparation des caractéristiques et de la cible
caracteristiques = df.drop('Type', axis=1)  # Caractéristiques
cible = df['Type']  # Cible (type de roche)

# Encodage des labels de la cible en valeurs numériques
encodeur_labels = LabelEncoder()
cible_encodee = encodeur_labels.fit_transform(cible)

# Division des données en ensembles d'entraînement et de test
caracteristiques_entrainement, caracteristiques_test, cible_entrainement, cible_test = train_test_split(
    caracteristiques, cible_encodee, test_size=0.2, random_state=42
)

# Création et entraînement du modèle de classification (arbre de décision)
modele = DecisionTreeClassifier(random_state=42)
modele.fit(caracteristiques_entrainement, cible_entrainement)

# Prédiction sur l'ensemble de test
cible_predite = modele.predict(caracteristiques_test)

# Évaluation du modèle
precision = accuracy_score(cible_test, cible_predite)
rapport_classification = classification_report(cible_test, cible_predite, target_names=encodeur_labels.classes_)

print(f'Précision : {precision}')
print(f'Rapport de Classification :\n{rapport_classification}')

def predire_type_roche(SiO2, Al2O3, Fe2O3, MgO, CaO, Na2O, K2O):
    """
    Fonction pour prédire le type de roche en fonction des caractéristiques chimiques.
    
    Paramètres :
    - SiO2 : Concentration de dioxyde de silicium
    - Al2O3 : Concentration d'alumine
    - Fe2O3 : Concentration d'oxyde de fer
    - MgO : Concentration d'oxyde de magnésium
    - CaO : Concentration d'oxyde de calcium
    - Na2O : Concentration d'oxyde de sodium
    - K2O : Concentration d'oxyde de potassium
    
    Retour :
    - Le type de roche prédit (Igneux, Sédimentaire, Métamorphique)
    """
    # Création d'un DataFrame avec les valeurs d'entrée
    nouvel_echantillon = pd.DataFrame({
        'SiO2': [SiO2],
        'Al2O3': [Al2O3],
        'Fe2O3': [Fe2O3],
        'MgO': [MgO],
        'CaO': [CaO],
        'Na2O': [Na2O],
        'K2O': [K2O]
    })
    
    # Prédiction
    type_predit_encode = modele.predict(nouvel_echantillon)
    type_predit = encodeur_labels.inverse_transform(type_predit_encode)
    return type_predit[0]

# Exemple d'utilisation
type_roche_predite = predire_type_roche(50, 15, 10, 5, 10, 3, 2)
print(f'Type de roche prédit : {type_roche_predite}')










import pandas as pd
import numpy as np

# Génération de données aléatoires
np.random.seed(42)
nombre_echantillons = 100
donnees = {
    'pH': np.random.uniform(6, 9, nombre_echantillons),  # pH de l'eau
    'Nitrates': np.random.uniform(0, 50, nombre_echantillons),  # Concentration en nitrates (mg/L)
    'Sulfates': np.random.uniform(0, 250, nombre_echantillons),  # Concentration en sulfates (mg/L)
    'Chlore': np.random.uniform(0, 5, nombre_echantillons),  # Concentration en chlore (mg/L)
    'Dureté': np.random.uniform(50, 300, nombre_echantillons),  # Dureté de l'eau (mg/L de CaCO3)
    'Qualité': np.random.choice(['Potable', 'Non potable'], nombre_echantillons)  # Cible
}

# Création du DataFrame
df = pd.DataFrame(donnees)

# Sauvegarde en fichier Excel
df.to_excel('donnees_qualite_eau.xlsx', index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Chargement du dataset
df = pd.read_excel('donnees_qualite_eau.xlsx')

# Séparation des caractéristiques et de la cible
caracteristiques = df.drop('Qualité', axis=1)  # Caractéristiques
cible = df['Qualité']  # Cible (qualité de l'eau)

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

def predire_qualite_eau(pH, nitrates, sulfates, chlore, durete):
    """
    Fonction pour prédire la qualité de l'eau en fonction des paramètres chimiques.
    
    Paramètres :
    - pH : pH de l'eau
    - nitrates : Concentration en nitrates (mg/L)
    - sulfates : Concentration en sulfates (mg/L)
    - chlore : Concentration en chlore (mg/L)
    - durete : Dureté de l'eau (mg/L de CaCO3)
    
    Retour :
    - La qualité de l'eau prédite (Potable ou Non potable)
    """
    # Création d'un DataFrame avec les valeurs d'entrée
    nouvel_echantillon = pd.DataFrame({
        'pH': [pH],
        'Nitrates': [nitrates],
        'Sulfates': [sulfates],
        'Chlore': [chlore],
        'Dureté': [durete]
    })
    
    # Prédiction
    qualite_predite_encode = modele.predict(nouvel_echantillon)
    qualite_predite = encodeur_labels.inverse_transform(qualite_predite_encode)
    return qualite_predite[0]

# Exemple d'utilisation
qualite_eau_predite = predire_qualite_eau(7.5, 10, 100, 2, 150)
print(f'Qualité de l\'eau prédite : {qualite_eau_predite}')





