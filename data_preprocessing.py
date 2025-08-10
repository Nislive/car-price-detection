import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):

    data = pd.read_csv(filepath)
    data = data.dropna()
    data["CarName"] = data["CarName"].str.strip().str.split().str[0]
    corrections = {
        "alfa-romero": "alfa-romeo",
        "maxda": "mazda",
        "porcshce": "porsche",
        "toyouta": "toyota",
        "vokswagen": "volkswagen",
        "vw": "volkswagen",
        "Nissan":"nissan"
    }
    data["CarName"]= data["CarName"].replace(corrections, regex=False)
    data = pd.get_dummies(data, columns=["CarName"], drop_first=True)

    data["fueltype"]= data["fueltype"].map({"gas":0, "diesel":1})
    data["aspiration"] = data["aspiration"].map({"std":0, "turbo":1})
    data["doornumber"] = data["doornumber"].map({"two":2, "four":4})
    data = pd.get_dummies(data, columns=["carbody"], drop_first=True)
    data = pd.get_dummies(data, columns=["drivewheel"], drop_first=True)

    mapping = {"four":4, "six":6, "five":5, "three":3, "twelve":12, "two":2, "eight":8}
    data["cylindernumber"] = data["cylindernumber"].map(mapping)

    X= data.drop(["car_ID", "symboling", "enginelocation", "wheelbase","carlength", "carwidth", "carheight", "enginetype", "fuelsystem", "boreratio", "stroke", "compressionratio", "price"], axis=1)
    Y= data.price
        
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    return X_train_scaled, X_test_scaled, Y_train, Y_test




