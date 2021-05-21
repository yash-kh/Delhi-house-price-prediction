import pickle
import json
import numpy as np
import pandas as pd

__locations = None
__data_columns = None
__model = None


def save_data(data):
    df1 = pd.read_csv("./artifacts/Data.csv")
    df2 = pd.DataFrame(data, columns=df1.columns)
    df1 = pd.concat([df1, df2])
    df1.to_csv("./artifacts/Data.csv", index=False)
    print('Saved')


def get_estimated_price(location, area, bhk, bath, ttype):
    try:
        loc_index = __data_columns.index(location.lower())

    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = area
    x[1] = bhk
    x[2] = bath
    if loc_index >= 0:
        x[loc_index] = 1
    if ttype == 'Apartment':
        x[37] = 1
    pred = round((__model.predict([x])[0])/100000, 2)
    data = np.array([[location, area, bhk, bath, ttype, pred]])
    save_data(data)

    return pred


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:-1]+['other']

    global __model
    if __model is None:
        with open('./artifacts/delhi_house_prices_model.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")


def get_location_names():
    return __locations


def get_data_columns():
    return __data_columns


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('Chhattarpur', 1175, 3, 3, 'Builder_Floor'))
    print(get_estimated_price('Sarita Vihar', 1500, 2, 3, 'Apartment'))
    print(get_estimated_price('Shahdara', 850, 3, 2, 'Builder_Floor'))
    print(get_estimated_price('other', 850, 3, 2, 'Builder_Floor'))
