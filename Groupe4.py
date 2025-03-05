import pandas as pd
import numpy as np

# Génération de données aléatoires
np.random.seed(42)
nombre_echantillons = 100
donnees = {
    'Quartz': np.random.uniform(10, 50, nombre_echantillons),  # Pourcentage de quartz
    'Feldspath': np.random.uniform(20, 60, nombre_echantillons),  # Pourcentage de feldspath
    'Mica': np.random.uniform(5, 30, nombre_echantillons),  # Pourcentage de mica
    'Amphibole': np.random.uniform(0, 20, nombre_echantillons),  # Pourcentage d'amphibole
    'Type_Roche': np.random.choice(['Igneux', 'Sédimentaire', 'Métamorphique'], nombre_echantillons)  # Cible
}

# Création du DataFrame
df = pd.DataFrame(donnees)

# Sauvegarde en fichier Excel
df.to_excel('donnees_composition_roches.xlsx', index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Chargement du dataset
df = pd.read_excel('donnees_composition_roches.xlsx')

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


def predire_type_roche(quartz, feldspath, mica, amphibole):
    """
    Fonction pour prédire le type de roche en fonction de la composition minéralogique.
    
    Paramètres :
    - quartz : Pourcentage de quartz
    - feldspath : Pourcentage de feldspath
    - mica : Pourcentage de mica
    - amphibole : Pourcentage d'amphibole
    
    Retour :
    - Le type de roche prédit (Igneux, Sédimentaire, Métamorphique)
    """
    # Création d'un DataFrame avec les valeurs d'entrée
    nouvel_echantillon = pd.DataFrame({
        'Quartz': [quartz],
        'Feldspath': [feldspath],
        'Mica': [mica],
        'Amphibole': [amphibole]
    })
    
    # Prédiction
    type_roche_predite_encode = modele.predict(nouvel_echantillon)
    type_roche_predite = encodeur_labels.inverse_transform(type_roche_predite_encode)
    return type_roche_predite[0]

# Exemple d'utilisation
type_roche_predite = predire_type_roche(30, 40, 15, 10)
print(f'Type de roche prédit : {type_roche_predite}')










import pandas as pd
import numpy as np

# Génération de données aléatoires
np.random.seed(42)
nombre_echantillons = 100
donnees = {
    'SiO2': np.random.uniform(40, 80, nombre_echantillons),  # Dioxyde de silicium
    'Al2O3': np.random.uniform(10, 20, nombre_echantillons),  # Alumine
    'FeO': np.random.uniform(1, 15, nombre_echantillons),  # Oxyde de fer
    'MgO': np.random.uniform(1, 10, nombre_echantillons),  # Oxyde de magnésium
    'Température': np.random.uniform(500, 1200, nombre_echantillons)  # Cible (température en °C)
}

# Création du DataFrame
df = pd.DataFrame(donnees)

# Sauvegarde en fichier Excel
df.to_excel('donnees_temperature_roches.xlsx', index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Chargement du dataset
df = pd.read_excel('donnees_temperature_roches.xlsx')

# Séparation des caractéristiques et de la cible
caracteristiques = df.drop('Température', axis=1)  # Caractéristiques
cible = df['Température']  # Cible (température)

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


def predire_temperature(SiO2, Al2O3, FeO, MgO):
    """
    Fonction pour prédire la température de formation des roches en fonction de leur composition chimique.
    
    Paramètres :
    - SiO2 : Concentration de dioxyde de silicium
    - Al2O3 : Concentration d'alumine
    - FeO : Concentration d'oxyde de fer
    - MgO : Concentration d'oxyde de magnésium
    
    Retour :
    - La température prédite (°C)
    """
    # Création d'un DataFrame avec les valeurs d'entrée
    nouvel_echantillon = pd.DataFrame({
        'SiO2': [SiO2],
        'Al2O3': [Al2O3],
        'FeO': [FeO],
        'MgO': [MgO]
    })
    
    # Prédiction
    temperature_predite = modele.predict(nouvel_echantillon)
    return temperature_predite[0]

# Exemple d'utilisation
temperature_predite = predire_temperature(60, 15, 8, 5)
print(f'Température prédite : {temperature_predite} °C')












import pandas as pd
import numpy as np

# Génération de données aléatoires
np.random.seed(42)
nombre_echantillons = 100
donnees = {
    'Densité': np.random.uniform(2, 5, nombre_echantillons),  # Densité (g/cm³)
    'Dureté': np.random.uniform(1, 10, nombre_echantillons),  # Dureté (échelle de Mohs)
    'SiO2': np.random.uniform(0, 100, nombre_echantillons),  # Pourcentage de SiO2
    'Minéral': np.random.choice(['Quartz', 'Feldspath', 'Mica', 'Amphibole'], nombre_echantillons)  # Cible
}

# Création du DataFrame
df = pd.DataFrame(donnees)

# Sauvegarde en fichier Excel
df.to_excel('donnees_mineraux.xlsx', index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Chargement du dataset
df = pd.read_excel('donnees_mineraux.xlsx')

# Séparation des caractéristiques et de la cible
caracteristiques = df.drop('Minéral', axis=1)  # Caractéristiques
cible = df['Minéral']  # Cible (minéral)

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


def predire_mineral(densite, durete, SiO2):
    """
    Fonction pour prédire le minéral en fonction de ses propriétés physiques et chimiques.
    
    Paramètres :
    - densite : Densité (g/cm³)
    - durete : Dureté (échelle de Mohs)
    - SiO2 : Pourcentage de SiO2
    
    Retour :
    - Le minéral prédit (Quartz, Feldspath, Mica, Amphibole)
    """
    # Création d'un DataFrame avec les valeurs d'entrée
    nouvel_echantillon = pd.DataFrame({
        'Densité': [densite],
        'Dureté': [durete],
        'SiO2': [SiO2]
    })
    
    # Prédiction
    mineral_predite_encode = modele.predict(nouvel_echantillon)
    mineral_predite = encodeur_labels.inverse_transform(mineral_predite_encode)
    return mineral_predite[0]

# Exemple d'utilisation
mineral_predite = predire_mineral(2.65, 7, 100)
print(f'Minéral prédit : {mineral_predite}')





