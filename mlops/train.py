import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def get_data():
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(URL, sep=";")
    df['quality'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)
    X = df.drop('quality', axis=1)
    y = df['quality']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train():
    X_train, X_test, y_train, y_test = get_data()

    # Bắt đầu một MLflow run
    with mlflow.start_run():
        print("Bắt đầu quá trình huấn luyện...")
        
        # Log các tham số
        params = {"solver": "liblinear", "random_state": 42}
        mlflow.log_params(params)
        
        # Huấn luyện mô hình
        lr = LogisticRegression(**params)
        lr.fit(X_train, y_train)
        
        # Đánh giá và log chỉ số
        y_pred = lr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"  Accuracy: {accuracy}")

        # Đăng ký mô hình vào MLflow Model Registry
        # Đây là bước quan trọng để quản lý phiên bản
        print("Đăng ký mô hình vào Model Registry...")
        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="model",
            registered_model_name="wine_quality_classifier" # Tên model trong Registry
        )
        print("Hoàn thành!")

if __name__ == "__main__":
    train()