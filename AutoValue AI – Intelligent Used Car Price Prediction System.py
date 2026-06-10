# Install Gradio library
!pip install gradio
import gradio as gr
import pandas as pd
import joblib


try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
except Exception as e:
    raise Exception(f"Error loading model files: {e}")


def predict_price(
    brand,
    fuel,
    transmission,
    owner,
    vehicle_age,
    kilometers,
    mileage,
    engine,
    power,
    seats
):
    try:

        data = {
            "Kilometers_Driven": kilometers,
            "Mileage": mileage,
            "Engine": engine,
            "Power": power,
            "Seats": seats,
            "Vehicle_Age": vehicle_age
        }

        input_df = pd.DataFrame([data])

      
        input_df[f"Fuel_Type_{fuel}"] = 1
        input_df[f"Transmission_{transmission}"] = 1

        if owner != "First":
            input_df[f"Owner_Type_{owner}"] = 1

        input_df[f"Brand_{brand}"] = 1

        
        for col in columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[columns]

        
        scaled_data = scaler.transform(input_df)

        
        prediction = model.predict(scaled_data)[0]

       
        health_score = max(
            0,
            min(
                100,
                int(
                    100
                    - (vehicle_age * 2)
                    - (kilometers / 5000)
                )
            )
        )

        
        if prediction < 5:
            category = "Budget Segment"
        elif prediction < 15:
            category = "Mid-Range Segment"
        else:
            category = "Premium Segment"

        return f"""
Estimated Price: ₹ {prediction:.2f} Lakh

Market Segment: {category}

Vehicle Health Score: {health_score}/100
"""

    except Exception as e:
        return f"Error: {str(e)}"



demo = gr.Interface(
    fn=predict_price,

    inputs=[
        gr.Dropdown(
            ["Maruti", "Hyundai", "Honda", "Toyota",
             "Ford", "Mahindra", "Tata", "BMW", "Audi"],
            label="Brand"
        ),

        gr.Dropdown(
            ["Petrol", "Diesel", "CNG"],
            label="Fuel Type"
        ),

        gr.Dropdown(
            ["Manual", "Automatic"],
            label="Transmission"
        ),

        gr.Dropdown(
            ["First", "Second", "Third"],
            label="Owner Type"
        ),

        gr.Slider(
            minimum=0,
            maximum=20,
            value=5,
            step=1,
            label="Vehicle Age (Years)"
        ),

        gr.Number(
            value=50000,
            label="Kilometers Driven"
        ),

        gr.Number(
            value=18,
            label="Mileage (kmpl)"
        ),

        gr.Number(
            value=1200,
            label="Engine (CC)"
        ),

        gr.Number(
            value=80,
            label="Power (BHP)"
        ),

        gr.Dropdown(
            [4, 5, 7],
            value=5,
            label="Number of Seats"
        )
    ],

    outputs=gr.Textbox(label="Prediction Result"),

    title="AutoValue AI",
    description="""
Intelligent Used Car Price Prediction System
Built using Machine Learning and Gradio.
""",

    examples=[
        ["Honda", "Petrol", "Manual", "First", 5, 50000, 18, 1200, 80, 5],
        ["BMW", "Diesel", "Automatic", "Second", 3, 30000, 15, 2500, 180, 5],
        ["Tata", "CNG", "Manual", "First", 2, 20000, 25, 1200, 90, 5]
    ]
)


demo.launch(share=True)
