# SpamClassifierAPI

SpamClassifierAPI is a machine learning project that classifies emails as either spam or ham (not spam). This project follows an MLOps approach, covering everything from data preprocessing, model training, hyperparameter tuning, to deploying the model using FastAPI to create an API that predicts whether a given email is spam or not.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [API Endpoints](#api-endpoints)
- [Related Work](#related-work)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started with the SpamClassifierAPI, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/SpamClassifierAPI.git
    cd SpamClassifierAPI
    ```

2. **Create and activate a virtual environment:**
    ```sh
    python -m venv spam
    spam\Scripts\activate (for Windows)
    source spam/bin/activate (for macOS/Linux)
    ```

3. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the FastAPI server:**
    ```sh
    uvicorn main:app --reload
    ```

## Usage

Once the server is running, you can interact with the API using HTTP requests. Here is an example using `curl` to make a POST request to the `/predict` endpoint:

```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "email": "Sample email content to classify."
}'
```

You should receive a JSON response indicating whether the email is classified as spam or ham.

## Project Structure

- `main.py`: Contains the FastAPI application and the endpoint definitions.
- `preprocess.py`: Contains the preprocessing code for the dataset.
- `spam-classifier.ipynb`: Jupyter notebook detailing the entire model training and evaluation process.
- `spam_classifier_model.pkl`: Serialized model file saved after training.
- `requirements.txt`: Lists the dependencies required for the project.

## Model Training and Evaluation

The `spam-classifier.ipynb` notebook contains the following steps:
1. **Data Preprocessing**: Loading the dataset, cleaning the text, removing stop words, and feature extraction.
2. **Model Training**: Training various models and selecting the best performing model.
3. **Hyperparameter Tuning**: Using RandomizedSearchCV to fine-tune the model hyperparameters.
4. **Ensemble Methods**: Combining the best models to improve performance.
5. **Model Evaluation**: Evaluating the model using cross-validation and test set.

## API Endpoints

- **`POST /predict`**: Predicts whether an email is spam or ham.
  - **Request**: 
    ```json
    {
      "email": "Sample email content to classify."
    }
    ```
  - **Response**:
    ```json
    {
      "prediction": "ham"
    }
    ```

## Related Work

- **Kaggle Notebook**: You can find the detailed Jupyter notebook for this project on [Kaggle](https://www.kaggle.com/code/liaichimustapha/spam-classifier).
- **Medium Article**: Read more about the project and its implementation on [Medium](https://medium.com/@mustaphaliaichi/end-to-end-machine-learning-project-113b39371801).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any bugs, feature requests, or improvements.

## License

This project is licensed under the MIT License.
