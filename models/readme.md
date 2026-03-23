# Bitcoin Forecast API

Projet de prevision du prix du Bitcoin a partir d'un historique en donnees minute.

Le depot contient :
- un pipeline de preparation des donnees temporelles
- plusieurs approches de prevision (`ARIMA`, `XGBoost`, `LSTM`)
- une API `FastAPI` qui expose des previsions reelles avec `ARIMA`, `XGBoost` et un mode `hybrid`

## Structure du projet

```text
Analyse_series_temporelles/
|-- api/
|   `-- app.py
|-- data/
|   `-- btcusd_1-min_data.csv
|-- models/
|   `-- readme.md
|-- notebooks/
|-- src/
|   |-- data_loader.py
|   |-- evaluate.py
|   |-- features.py
|   |-- predictor.py
|   |-- preprocessing.py
|   |-- train_arima.py
|   |-- train_lstm.py
|   `-- train_xgboost.py
`-- requirements.txt
```

## Dataset

Le projet utilise le fichier :

`data/btcusd_1-min_data.csv`

Colonnes attendues :
- `Timestamp`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

## Pipeline

Le flux principal est le suivant :

1. Chargement du CSV avec `src/data_loader.py`
2. Conversion du timestamp en date dans `src/preprocessing.py`
3. Reechantillonnage journalier et creation de variables derivees dans `src/features.py`
4. Entrainement / prevision via :
   - `src/train_arima.py`
   - `src/train_xgboost.py`
   - `src/train_lstm.py`
5. Exposition des previsions par l'API dans `api/app.py`

## Modeles

### ARIMA

Modele statistique de serie temporelle applique a la serie `Log_Price`.

Usage actuel :
- entrainement / evaluation dans `src/train_arima.py`
- prevision API disponible via `model=arima`

### XGBoost

Modele de regression sur variables construites a partir des prix precedents :
- retards `t-1`, `t-2`
- moyenne mobile `SMA_7`
- ecart-type mobile `Std_7`
- variation moyenne recente
- jour de la semaine

Usage actuel :
- entrainement / evaluation dans `src/train_xgboost.py`
- prevision API disponible via `model=xgboost`

### LSTM

Modele sequence-to-one sur la serie `Close` normalisee.

Usage actuel :
- entrainement / evaluation dans `src/train_lstm.py`
- non expose dans l'API pour le moment

## API

L'API FastAPI se trouve dans :

`api/app.py`

### Lancer l'API

```powershell
.\venv\Scripts\python.exe -m uvicorn api.app:app --reload
```

### Endpoints disponibles

#### `GET /`

Retourne les informations de base de l'API.

#### `GET /health`

Verifie que le service repond.

Exemple de reponse :

```json
{
  "status": "ok"
}
```

#### `GET /predict`

Genere une prevision reelle a partir du dataset local.

Parametres :
- `days` : nombre de jours a predire, entre `1` et `30`
- `model` : `xgboost`, `arima` ou `hybrid`

Exemple :

```text
/predict?days=7&model=hybrid
```

Exemple de reponse :

```json
{
  "forecast_days": 7,
  "model": "hybrid",
  "latest_observation_date": "2026-03-05",
  "latest_close": 50615.32,
  "predictions": [
    {
      "date": "2026-03-06",
      "close": 51102.81
    },
    {
      "date": "2026-03-07",
      "close": 51699.93
    }
  ],
  "rmse": {
    "xgboost": 1234.56,
    "arima": 1456.78
  }
}
```

Remarque :
- les valeurs numeriques ci-dessus sont un exemple de format
- les previsions reelles dependent du contenu actuel du fichier CSV

## Installation

Installer les dependances :

```powershell
pip install -r requirements.txt
```

Dependances principales :
- `pandas`
- `numpy`
- `scikit-learn`
- `statsmodels`
- `xgboost`
- `tensorflow`
- `fastapi`
- `uvicorn`

## Etat actuel du projet

Ce qui fonctionne :
- preparation des donnees
- entrainement local ARIMA / XGBoost / LSTM
- API de prediction avec `ARIMA`, `XGBoost` et `hybrid`

Ce qui n'est pas encore en place :
- sauvegarde persistante des modeles dans des fichiers `.pkl` ou `.h5`
- endpoint API base sur `LSTM`
- pipeline de tests automatise
- documentation racine `README.md`

## Notes

- Le premier appel a `/predict` peut etre plus lent car il charge le gros CSV et calcule les previsions.
- L'API utilise le fichier local `data/btcusd_1-min_data.csv`.
- Le mode `hybrid` moyenne actuellement les sorties `ARIMA` et `XGBoost`.
