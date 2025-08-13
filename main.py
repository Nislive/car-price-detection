from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from data_preprocessing import load_and_preprocess_data

X_train, X_test, Y_train, Y_test = load_and_preprocess_data("./data/raw_data.csv")

model = linear_model.LinearRegression()

model.fit(X_train, Y_train)

