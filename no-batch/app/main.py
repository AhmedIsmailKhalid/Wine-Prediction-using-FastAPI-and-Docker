# Importing the neccessary libraries
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Create instance of FastAPI with title
app = FastAPI(title="Predicting Wine Class")

# Creating a class the subclasses from BaseModel. This is done to represent a data point (in this case a particular wine) and listing each attribute along with its corresponding type
class Wine(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

# Load the classifier into memory for prediction. Decorate the function to run at startup of server
@app.on_event("startup")
def load_clf():
    # Load classifier from pickle file
    with open("/app/wine.pkl", "rb") as file:
        global clf
        clf = pickle.load(file)

# Create a function for homepage for testing i
@app.get("/")
def home():
    return "The API is working as expected. Visit http://localhost:80/docs for more"

# Creat function to serve predictions. This fucntion will be called/used when visiting /predict endpoint
@app.post("/predict")
def predict(wine: Wine):
    data_point = np.array(
        [
            [
                wine.alcohol,
                wine.malic_acid,
                wine.ash,
                wine.alcalinity_of_ash,
                wine.magnesium,
                wine.total_phenols,
                wine.flavanoids,
                wine.nonflavanoid_phenols,
                wine.proanthocyanins,
                wine.color_intensity,
                wine.hue,
                wine.od280_od315_of_diluted_wines,
                wine.proline,
            ]
        ]
    )

    pred = clf.predict(data_point).tolist()
    pred = pred[0]
    print(pred)
    return {"Prediction": pred}
