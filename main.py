from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from data_preprocessing import load_and_preprocess_data
import seaborn as sns
import matplotlib.pyplot as plt

X_train, X_test, Y_train, Y_test = load_and_preprocess_data("./data/raw_data.csv")

model = linear_model.LinearRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)


mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"Mean squared error (MSE): {mse:.2f}")
print(f"Coefficient of determination (R^2): {r2:.2f}")

sns.scatterplot(x=Y_test, y=Y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color="red")
plt.savefig("results.png", dpi=300, bbox_inches="tight")

