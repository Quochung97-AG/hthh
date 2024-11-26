import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def importdata():
    df = pd.read_csv("iris.csv")
    # use required features
    cdf = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width','variety']]

    return df, cdf

def predictor(cdf):
    # Encode target variable (variety) to numeric
    label_encoder = LabelEncoder()
    cdf['variety'] = label_encoder.fit_transform(cdf['variety'])

    # Training Data and Predictor Variable
    # Use all data for training (tarin-test-split not used)
    x = cdf.iloc[:, 0:4]
    y = cdf.iloc[:, 4]
    regressor = LinearRegression()

    # Fitting model with trainig data
    regressor.fit(x, y)

    # Saving model to current directory
    # Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
    model_data = {"model":regressor,"label_encoder":label_encoder}
    pickle.dump(model_data, open('model.pkl', 'wb'))
    print("Mô hình đã Label_encoder được huấn luyện và lưu thành 'model.pkl'.")

def make_prediction():
    # Kiểm tra sự tồn tại của model.pkl
    if not os.path.exists('model.pkl'):
        print("File 'model.pkl' không tồn tại. Vui lòng kiểm tra.")
        return

    # Thử tải mô hình từ file
    try:
        model_data = pickle.load(open('model.pkl', 'rb'))
        loaded_model = model_data["model"]
        label_encoder = model_data["label_encoder"]
    except Exception as e:
        print("Không thể tải mô hình:", e)
        return
    # Loading model to compare the results

    sample_data = pd.DataFrame([[5.9, 3, 4.2, 1.5]], columns=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'])
    predicted_variety_numberic = loaded_model.predict(sample_data)
    predicted_variety = label_encoder.inverse_transform([int(round(predicted_variety_numberic[0]))])
    print(f"Dự đoán loài iris cho {sample_data.values}: {predicted_variety[0]}")


def hoadv():
    df, cdf = importdata()
    predictor(cdf)
    make_prediction()

if __name__=="__main__":
    hoadv()