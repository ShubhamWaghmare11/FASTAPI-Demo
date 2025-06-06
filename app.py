from fastapi import FastAPI
from fastapi.responses import JSONResponse
from schema.user_input import UserInput
from model.predict import predict_output, model, MODEL_VERSION
from schema.prediction_response import PredictionResponse
app = FastAPI()


@app.get("/")
def home():
    return {'message':"Insurance Premium Prediction API"}


@app.get("/health")
def health_check():
    return {
        "status":"OK",
        "version":MODEL_VERSION,
        'model_loaded': model is not None

    }

@app.post("/predict", response_model=PredictionResponse)
def predict_premium(data: UserInput):
    user_input = data.model_dump(exclude=['age', 'weight', 'height', 'smoker', 'city'])

    try:
        prediction,confidence,class_probs = predict_output(user_input)

        return JSONResponse(status_code=200,content={"response":{"predicted_category":str(prediction[0]),
                                                     "confidence":confidence,
                                                     "class_probabilities":class_probs}})
    
    except Exception as e:
        return JSONResponse(status_code=500,content=str(e))