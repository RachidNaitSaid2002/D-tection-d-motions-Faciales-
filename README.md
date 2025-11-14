# Détection d’Émotions Faciales

Ce projet est un système de **détection d'émotions faciales** (intitulé "D-tection-d-motions-Faciales-" dans le README, probablement "Détection d’Émotions Faciales" en français, signifiant "Facial Emotion Detection"). Il intègre l'apprentissage automatique pour la reconnaissance des émotions avec un backend API web pour des prédictions en temps réel, le stockage en base de données et les tests automatisés.

### Structure du Projet

| Répertoire/Fichier | Description |
|--------------------|-------------|
| `ML/` | Module d'apprentissage automatique |
| ├── `NoteBook/NoteBook.ipynb` | Notebook Jupyter démontrant le pipeline complet (chargement des données, entraînement, tests) |
| ├── `Prediction/Pridection_func.py` | Fonction de prédiction utilisant les cascades Haar (OpenCV) |
| ├── `MtCNN/Pridection_func.py` | Fonction de prédiction utilisant MTCNN pour la détection de visages |
| ├── `Haarcascade/haarcascade_frontalface_default.xml` | Fichier XML pour les cascades Haar |
| └── `Model/Model.dump` | Modèle CNN entraîné sauvegardé avec Joblib |
| `Backend/` | API backend |
| ├── `main.py` | Application FastAPI principale avec endpoints |
| ├── `DB/database.py` | Configuration de la base de données PostgreSQL |
| ├── `models/Predictions.py` | Modèle SQLAlchemy pour la table Predictions |
| └── `schemas/prediction.py` | Schémas Pydantic pour la validation des données |
| `.github/workflows/python-tests.yml` | Workflow GitHub Actions pour les tests CI |
| `test_unit.py` | Tests unitaires avec Pytest |
| `requirements.txt` | Liste des dépendances Python |
| `.gitignore` | Fichiers et répertoires ignorés par Git |
| `README.md` | Ce fichier de documentation |
| `.env` | Variables d'environnement (non versionné) |
| `PredictiosResults/` | Répertoire pour les images de prédictions traitées |
| `images_Test/` | Images de test pour les prédictions |
| `Data/` | Données d'entraînement et de test (train/test) (https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data) |

### Composants Clés :
- **Module d'Apprentissage Automatique (ML)** :
  - Un modèle de réseau de neurones convolutifs (CNN) entraîné en utilisant TensorFlow/Keras sur un ensemble de données d'images faciales (probablement à partir de répertoires comme `../../Data/train` et `../../Data/test`).
  - Le modèle classe les visages en 7 émotions : en colère, dégoûté, effrayé, heureux, neutre, triste, surpris.
  - La détection de visages utilise deux méthodes :
    - Cascades Haar (via OpenCV) dans `ML/Prediction/Pridection_func.py`.
    - MTCNN (Multi-Task Cascaded Convolutional Networks) dans `ML/MtCNN/Pridection_func.py`.
  - Les images sont redimensionnées à 32x32 pixels pour la prédiction. Le modèle entraîné est sauvegardé/chargé en utilisant Joblib (`ML/Model/Model.dump`).
  - Le notebook Jupyter (`ML/NoteBook/NoteBook.ipynb`) démontre le pipeline complet : chargement des données, prétraitement (normalisation, encodage one-hot), augmentation des données, entraînement du modèle (avec arrêt précoce), évaluation, et tests en direct sur des images/vidéos via webcam.

- **API Backend** :
  - Construit avec FastAPI (`Backend/main.py`).
  - Endpoints :
    - `GET /` : Endpoint racine retournant un message "Hello World".
    - `POST /Prediction` : Accepte les téléchargements d'images, les traite pour la détection d'émotions, stocke les résultats dans la base de données, et retourne les détails de prédiction (étiquette d'émotion, score, ID).
    - `GET /Prediction` : Récupère toutes les prédictions stockées.
  - Utilise SQLAlchemy pour ORM et Pydantic pour la validation des données/schémas.
  - Base de données : PostgreSQL (configuré via variables d'environnement dans `.env`).
  - Les prédictions sont sauvegardées avec des métadonnées (émotion, score, chemin d'image) et les images traitées sont stockées dans `PredictiosResults/{id}/Image.jpg`.

- **Base de Données** :
  - Configuration PostgreSQL dans `Backend/DB/database.py`.
  - Modèle : Table `Predictions` avec colonnes pour ID, étiquette de prédiction, score, et chemin d'image.

- **Tests et CI** :
  - Tests unitaires dans `test_unit.py` utilisant Pytest : Vérifie le chargement du modèle et le format de réponse de l'API de prédiction.
  - Workflow GitHub Actions (`.github/workflows/python-tests.yml`) exécute les tests sur push vers `main`, configurant PostgreSQL et installant les dépendances.

- **Dépendances** (de `requirements.txt`) :
  - Core : joblib, fastapi, tensorflow, numpy, matplotlib, opencv-python, sqlalchemy, pydantic, mtcnn, pytest, httpx, dotenv, python-multipart.

- **Fichiers Ignorés** (`.gitignore`) :
  - Fichiers d'environnement, répertoires de cache, dossiers de données, et résultats de prédiction.

### Installation et Exécution

#### Prérequis
- Python 3.11 ou supérieur
- PostgreSQL (pour la base de données)
- Git

#### Téléchargement du Projet
Clonez le dépôt depuis GitHub :
```bash
git clone https://github.com/votre-utilisateur/D-tection-d-motions-Faciales-.git
cd D-tection-d-motions-Faciales-
```

#### Configuration de l'Environnement
1. Créez un environnement virtuel :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows : venv\Scripts\activate
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Configurez la base de données :
   - Créez une base de données PostgreSQL.
   - Créez un fichier `.env` à la racine du projet avec les variables d'environnement suivantes :
     ```
     user=votre_utilisateur_postgres
     password=votre_mot_de_passe
     host=localhost
     port=5432
     database=votre_base_de_donnees
     ```

#### Exécution du Projet
1. Lancez l'API backend :
   ```bash
   cd Backend
   python main.py
   ```
   L'API sera disponible sur `http://localhost:8000`.

2. Pour tester l'API :
   - Endpoint racine : `GET http://localhost:8000/`
   - Prédiction : `POST http://localhost:8000/Prediction` (avec une image en multipart/form-data)
   - Récupérer les prédictions : `GET http://localhost:8000/Prediction`

3. Pour exécuter les tests :
   ```bash
   pytest test_unit.py
   ```

4. Pour explorer le notebook ML :
   - Ouvrez `ML/NoteBook/NoteBook.ipynb` dans Jupyter Notebook ou JupyterLab.

### Fonctionnalité Globale :
- Les utilisateurs peuvent télécharger des images via l'API pour détecter les émotions dans les visages.
- Prend en charge à la fois les images statiques et potentiellement la vidéo en direct (démontré dans le notebook).
- Les résultats sont persistés dans une base de données avec des images traitées sauvegardées.
- Le système est conçu pour le déploiement avec CI/CD et tests.
