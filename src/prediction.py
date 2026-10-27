import pandas as pd


def predict_car_price(model, car_details):
    """Predict a used car's price from its input features."""
    input_data = pd.DataFrame([car_details])

    predicted_price = model.predict(input_data)[0]

    return max(0.0, float(predicted_price))
