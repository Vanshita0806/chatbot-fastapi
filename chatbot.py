from fastapi import FastAPI, Request
import numpy as np
import json
from typing import List
import pickle

app = FastAPI()

with open("disease_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("symptoms_dict.pkl", "rb") as file:
    symptoms_dict = pickle.load(file)

# Load diseases list dictionary
with open("diseases_list.pkl", "rb") as file:
    diseases_list = pickle.load(file)


# Example disease prediction function (same as your existing one)
def predict_disease(symptoms: List[str]):
    patient_symptoms = [symptom.lower().replace(" ", "_") for symptom in symptoms]
    input_vector = np.zeros(len(symptoms_dict))  # symptoms_dict should be preloaded

    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
        else:
            print(f"Warning: Symptom '{symptom}' not found in training data.")

    predicted_probs = model.predict_proba([input_vector])[0]

    predicted_diseases = {
        diseases_list[label]: prob
        for label, prob in enumerate(predicted_probs) if prob > 0.01
    }

    sorted_diseases = sorted(predicted_diseases.items(), key=lambda x: x[1], reverse=True)[:5]

    return sorted_diseases

@app.post("/webhook")
async def dialogflow_webhook(request: Request):
    req_data = await request.json()

    # Extract intent name
    intent = req_data["queryResult"]["intent"]["displayName"]

    # If intent is "Predict Disease", process symptoms
    if intent == "Predict Disease":
        symptoms_text = req_data["queryResult"]["parameters"]["symptoms"]
        symptoms_list = [s.strip() for s in symptoms_text.split(",")]

        predicted_diseases = predict_disease(symptoms_list)

        # Create a response for Dialogflow
        if predicted_diseases:
            response_text = "Based on your symptoms, you might have: " + ", ".join([d[0] for d in predicted_diseases])
        else:
            response_text = "I couldn't identify a disease based on the symptoms provided."

        return {
            "fulfillmentText": response_text
        }

    return {
        "fulfillmentText": "I am not sure how to handle this request."
    }
