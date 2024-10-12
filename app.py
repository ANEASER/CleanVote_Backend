from fastapi import FastAPI
import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = FastAPI()

# Load the pre-trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the data
df = pd.read_csv('./PE_All.csv')
df = df.tail(500)

@app.get("/predict")
def predict():
    # Filter the dataset for the 2024 election
    df_2024_predict = df[df['election_year'] == 2024]

    # Select features and labels
    X = df[['election_year', 'postal_data', 'poll_data', 'final_percentage',
            'negative_prob', 'neutral_prob', 'positive_prob']]
    y = df['percentage']

    # Filter the test data for 2024
    X_test = X[X['election_year'] == 2024]
    y_test = y[X['election_year'] == 2024]

    # Make predictions
    y_pred = model.predict(X_test)

    # Add predicted percentages to the dataframe
    df_2024_predict['predicted_percentage'] = y_pred

    # Group the results by candidate and calculate mean predicted percentage
    final_result = df_2024_predict.groupby('candidate')['predicted_percentage'].mean().reset_index()

    # Return the final result as JSON
    return final_result.to_dict(orient='records')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
