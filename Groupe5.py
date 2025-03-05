import pandas as pd
import numpy as np

# Génération de données aléatoires
np.random.seed(42)
nombre_echantillons = 100
donnees = {
    'Altitude': np.random.uniform(0, 1000, nombre_echantillons),  # Altitude (m)
    'Pente': np.random.uniform(0, 30, nombre_echantillons),  # Pente (degrés)
    'NDVI': np.random.uniform(-1, 1, nombre_echantillons),  # Indice de végétation
    'Type_Sol': np.random.choice(['Argileux', 'Sableux', 'Limoneux'], nombre_echantillons)  # Cible
}

# Création du DataFrame
df = pd.DataFrame(donnees)

# Sauvegarde en fichier Excel
df.to_excel('donnees_types_sols.xlsx', index=False)



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Chargement du dataset
df = pd.read_excel('donnees_types_sols.xlsx')

# Séparation des caractéristiques et de la cible
caracteristiques = df.drop('Type_Sol', axis=1)  # Caractéristiques
cible = df['Type_Sol']  # Cible (type de sol)

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



def predire_type_sol(altitude, pente, ndvi):
    """
    Fonction pour prédire le type de sol en fonction des données géospatiales.
    
    Paramètres :
    - altitude : Altitude (m)
    - pente : Pente (degrés)
    - ndvi : Indice de végétation (NDVI)
    
    Retour :
    - Le type de sol prédit (Argileux, Sableux, Limoneux)
    """
    # Création d'un DataFrame avec les valeurs d'entrée
    nouvel_echantillon = pd.DataFrame({
        'Altitude': [altitude],
        'Pente': [pente],
        'NDVI': [ndvi]
    })
    
    # Prédiction
    type_sol_predite_encode = modele.predict(nouvel_echantillon)
    type_sol_predite = encodeur_labels.inverse_transform(type_sol_predite_encode)
    return type_sol_predite[0]

# Exemple d'utilisation
type_sol_predite = predire_type_sol(500, 15, 0.5)
print(f'Type de sol prédit : {type_sol_predite}')








import pandas as pd
import numpy as np

# Génération de données aléatoires
np.random.seed(42)
nombre_echantillons = 100
donnees = {
    'Altitude': np.random.uniform(0, 1000, nombre_echantillons),  # Altitude (m)
    'Pente': np.random.uniform(0, 30, nombre_echantillons),  # Pente (degrés)
    'NDVI': np.random.uniform(-1, 1, nombre_echantillons),  # Indice de végétation
    'Utilisation_Terres': np.random.choice(['Agricole', 'Urbaine', 'Forestière'], nombre_echantillons)  # Cible
}

# Création du DataFrame
df = pd.DataFrame(donnees)

# Sauvegarde en fichier Excel
df.to_excel('donnees_utilisation_terres.xlsx', index=False)



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Chargement du dataset
df = pd.read_excel('donnees_utilisation_terres.xlsx')

# Séparation des caractéristiques et de la cible
caracteristiques = df.drop('Utilisation_Terres', axis=1)  # Caractéristiques
cible = df['Utilisation_Terres']  # Cible (utilisation des terres)

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

def predire_utilisation_terres(altitude, pente, ndvi):
    """
    Fonction pour prédire l'utilisation des terres en fonction des données géospatiales.
    
    Paramètres :
    - altitude : Altitude (m)
    - pente : Pente (degrés)
    - ndvi : Indice de végétation (NDVI)
    
    Retour :
    - L'utilisation des terres prédite (Agricole, Urbaine, Forestière)
    """
    # Création d'un DataFrame avec les valeurs d'entrée
    nouvel_echantillon = pd.DataFrame({
        'Altitude': [altitude],
        'Pente': [pente],
        'NDVI': [ndvi]
    })
    
    # Prédiction
    utilisation_predite_encode = modele.predict(nouvel_echantillon)
    utilisation_predite = encodeur_labels.inverse_transform(utilisation_predite_encode)
    return utilisation_predite[0]

# Exemple d'utilisation
utilisation_predite = predire_utilisation_terres(300, 10, 0.2)
print(f'Utilisation des terres prédite : {utilisation_predite}')









import pandas as pd
import numpy as np

# Génération de données aléatoires
np.random.seed(42)
nombre_echantillons = 100
donnees = {
    'Densité_Population': np.random.uniform(0, 10000, nombre_echantillons),  # Densité de population (habitants/km²)
    'Proximité_Routes': np.random.uniform(0, 10, nombre_echantillons),  # Proximité des routes (km)
    'NDVI': np.random.uniform(-1, 1, nombre_echantillons),  # Indice de végétation
    'Qualité_Air': np.random.choice(['Bonne', 'Moyenne', 'Mauvaise'], nombre_echantillons)  # Cible
}

# Création du DataFrame
df = pd.DataFrame(donnees)

# Sauvegarde en fichier Excel
df.to_excel('donnees_qualite_air.xlsx', index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Chargement du dataset
df = pd.read_excel('donnees_qualite_air.xlsx')

# Séparation des caractéristiques et de la cible
caracteristiques = df.drop('Qualité_Air', axis=1)  # Caractéristiques
cible = df['Qualité_Air']  # Cible (qualité de l'air)

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


def predire_qualite_air(densite_population, proximite_routes, ndvi):
    """
    Fonction pour prédire la qualité de l'air en fonction des données géospatiales.
    
    Paramètres :
    - densite_population : Densité de population (habitants/km²)
    - proximite_routes : Proximité des routes (km)
    - ndvi : Indice de végétation (NDVI)
    
    Retour :
    - La qualité de l'air prédite (Bonne, Moyenne, Mauvaise)
    """
    # Création d'un DataFrame avec les valeurs d'entrée
    nouvel_echantillon = pd.DataFrame({
        'Densité_Population': [densite_population],
        'Proximité_Routes': [proximite_routes],
        'NDVI': [ndvi]
    })
    
    # Prédiction
    qualite_air_predite_encode = modele.predict(nouvel_echantillon)
    qualite_air_predite = encodeur_labels.inverse_transform(qualite_air_predite_encode)
    return qualite_air_predite[0]

# Exemple d'utilisation
qualite_air_predite = predire_qualite_air(5000, 5, 0.3)
print(f'Qualité de l\'air prédite : {qualite_air_predite}')

