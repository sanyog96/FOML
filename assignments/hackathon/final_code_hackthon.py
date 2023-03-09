import time
import numpy as np
import pandas as pd
import warnings
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import category_encoders as ce
warnings.filterwarnings("ignore")

train_df = pd.read_csv("./train.csv", parse_dates=["Crash Date/Time"])
test_df = pd.read_csv("./test.csv", parse_dates=["Crash Date/Time"])
test_uniq_id = test_df["Id"]
test_df = test_df.drop(["Id"], axis=1)

map_dict = dict()

useless_col = ["Road Name", "Cross-Street Name", "Off-Road Description", "Related Non-Motorist", "Person ID"
,"Vehicle ID", "Vehicle Continuing Dir", "Vehicle Going Dir", "Vehicle Make", "Vehicle Model", "Location", "Crash Date/Time"]

null_val_cols = ['Local Case Number', 'Route Type', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 
'Off-Road Description', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Weather', 'Surface Condition', 
'Light', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Circumstance', 'Drivers License State', 
'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 
'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Vehicle Make', 'Vehicle Model', 'Equipment Problems']

map_dict["Agency Name"] = {"Montgomery County Police":9,"MONTGOMERY":8,"Rockville Police Departme":7,"Gaithersburg Police Depar":6
,"Takoma Park Police Depart":5,"Maryland-National Capital":4,"ROCKVILLE":3,"GAITHERSBURG":2,"MCPARK":1,"TAKOMA":0}
map_dict["ACRS Report Type"] = {'Property Damage Crash':2,'Injury Crash':1,'Fatal Crash':0}
map_dict["Route Type"] = {"Maryland (State)":9,"County":8,"Municipality":7,"US (State)":6,"Interstate (State)":5,"Other Public Roadway":4,"Government":3,"Ramp":2,"Service Road":1,"Unknown":0}
map_dict["Cross-Street Type"] = {"County":9,"Maryland (State)":8,"Municipality":7,"Unknown":6,"Ramp":5,"Other Public Roadway":4,"US (State)":3,"Government":2,"Interstate (State)":1,"Service Road":0}
map_dict["Collision Type"] = {"SAME DIR REAR END":17,"STRAIGHT MOVEMENT ANGLE":16,"SINGLE VEHICLE":15,"OTHER":14,"SAME DIRECTION SIDESWIPE":13,"HEAD ON LEFT TURN":12,"HEAD ON":11
,"SAME DIRECTION RIGHT TURN":10,"SAME DIRECTION LEFT TURN":9,"OPPOSITE DIRECTION SIDESWIPE":8,"ANGLE MEETS LEFT TURN":7,"ANGLE MEETS RIGHT TURN":6,"UNKNOWN":5
,"SAME DIR REND RIGHT TURN":4,"SAME DIR REND LEFT TURN":3,"ANGLE MEETS LEFT HEAD ON":2,"SAME DIR BOTH LEFT TURN":1,"OPPOSITE DIR BOTH LEFT TURN":0}
map_dict["Weather"] = {"CLEAR":11,"RAINING":10,"CLOUDY":9,"SNOW":8,"UNKNOWN":7,"FOGGY":6,"WINTRY MIX":5,"OTHER":4,"SLEET":3,"SEVERE WINDS":2,"BLOWING SNOW":1,"BLOWING SAND, SOIL, DIRT":0}
map_dict["Light"] = {"DAYLIGHT":8,"DARK LIGHTS ON":7,"DARK NO LIGHTS":6,"DUSK":5,"DAWN":4,"DARK -- UNKNOWN LIGHTING":3,"UNKNOWN":2,"OTHER":1}
map_dict["Surface Condition"] = {"DRY":10,"WET":9,"ICE":8,"SNOW":7,"UNKNOWN":6,"SLUSH":5,"WATER(STANDING/MOVING)":4,"MUD, DIRT, GRAVEL":3,"OIL":2,"OTHER":1,"SAND":0}
map_dict["Traffic Control"] = {"NO CONTROLS":10,"TRAFFIC SIGNAL":9,"STOP SIGN":8,"OTHER":7,"FLASHING TRAFFIC SIGNAL":6
,"YIELD SIGN":5,"PERSON":4,"WARNING SIGN":3,"UNKNOWN":2,"RAILWAY CROSSING DEVICE":1,"SCHOOL ZONE SIGN DEVICE":0}
map_dict["Driver Substance Abuse"] = {"NONE DETECTED":0,"UNKNOWN":1,"ALCOHOL PRESENT":1,"ALCOHOL CONTRIBUTED":1,"ILLEGAL DRUG PRESENT":1,"MEDICATION PRESENT":1
,"ILLEGAL DRUG CONTRIBUTED":1,"COMBINED SUBSTANCE PRESENT":1,"OTHER":1,"MEDICATION CONTRIBUTED":1,"COMBINATION CONTRIBUTED":1}
map_dict["Injury Severity"] = {'NO APPARENT INJURY':0,'POSSIBLE INJURY':1,'SUSPECTED MINOR INJURY':2,
                               'SUSPECTED SERIOUS INJURY':3,'FATAL INJURY':4}
map_dict["Vehicle Damage Extent"] = {'OTHER':1,'NO DAMAGE':0,'SUPERFICIAL':2,'FUNCTIONAL':3,'DISABLING':4,'DESTROYED':5,
                                    'UNKNOWN':1}
map_dict["Vehicle First Impact Location"] = {"TWELVE OCLOCK":16,"ONE OCLOCK":15,"ELEVEN OCLOCK":14,"SIX OCLOCK":13,"TWO OCLOCK":12,"FIVE OCLOCK":11
,"FOUR OCLOCK":10,"UNKNOWN":9,"TEN OCLOCK":8,"THREE OCLOCK":7,"SEVEN OCLOCK":6,"NINE OCLOCK":5,"EIGHT OCLOCK":4 
,"UNDERSIDE":3,"ROOF TOP":2,"NON-COLLISION":1}
map_dict["Vehicle Second Impact Location"] = {"TWELVE OCLOCK":16,"ONE OCLOCK":15,"ELEVEN OCLOCK":14,"SIX OCLOCK":13,"TWO OCLOCK":12
,"UNKNOWN":11,"FOUR OCLOCK":10,"FIVE OCLOCK":9,"TEN OCLOCK":8,"THREE OCLOCK":7,"SEVEN OCLOCK":6
,"NINE OCLOCK":5,"EIGHT OCLOCK":4,"UNDERSIDE":3,"ROOF TOP":2,"NON-COLLISION":1}
map_dict["Driverless Vehicle"] = {"Unknown":0, "No":1}
map_dict["Parked Vehicle"] = {"Yes":0, "No":1}
map_dict["Equipment Problems"] = {"NO MISUSE":0,"UNKNOWN":1,"OTHER":1,"BELTS/ANCHORS BROKE":1,"AIR BAG FAILED":1,
                                 "SIZE/TYPE IMPROPER":1,"NOT STREPPED RIGHT":1,"STRAP/TETHER LOOSE":1,"BELT(S) MISUSED":1,"FACING WRONG WAY":1}


def cleandata(X):
    #for col in label_encoder_col:
    #    le = preprocessing.LabelEncoder()
    #    X[col] = le.fit_transform((X[col]))
    
    #X = X.drop(label_encoder_col +binary_encoder_col, axis=1)
    #encoder = ce.BinaryEncoder(cols=binary_encoder_col)
    #X = encoder.fit_transform(X)
    
    X["Local Case Number"] = X["Local Case Number"].str.extract('(\d+)', expand=False).astype(float)
    
    for col in ["Vehicle Body Type", "Vehicle Movement", "Drivers License State", "Report Number"]:
        values = X[col].value_counts().keys().tolist()
        counts = [(n/X.shape[0]) for n in X[col].value_counts().tolist()]
        k = {values[i]: counts[i] for i in range(len(values))}
        X[col].replace(k, inplace=True)

    for key in map_dict:
        X[key].replace(map_dict[key], inplace=True)

    for col in null_val_cols:
        X[col] = X[col].fillna(-1)
    
    for col in ["Non-Motorist Substance Abuse", "Circumstance", "Municipality"]:
        temp = X[col]
        temp[~(temp.isnull())] = 0
        temp[temp.isnull()] = 1
        X[col] = temp
    
    X["direction"] = np.where(X["Vehicle Continuing Dir"] == X["Vehicle Going Dir"], 0, 1)
    
    X["year"] = X["Crash Date/Time"].dt.year
    X["month"] = X["Crash Date/Time"].dt.month
    X["day"] = X["Crash Date/Time"].dt.day
    X["hour"] = X["Crash Date/Time"].dt.hour
    X["dayofweek"] = X["Crash Date/Time"].dt.dayofweek
    X['dayofyear'] = X["Crash Date/Time"].dt.dayofyear
    X["quarter"] = X["Crash Date/Time"].dt.quarter
    
    X['Location'].replace(',', ' ', regex=True, inplace=True)
    X[['Location_x', 'Location_y']] = X['Location'].str.split(' ', expand=True)
    
    X = X.drop(useless_col, axis=1)
    
    return X

X_train = cleandata(train_df)
X_train = X_train.drop(["Fault"], axis=1)
y_train = np.ravel(train_df["Fault"])
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=5)
print(X_train.shape)
X_test = cleandata(test_df)
print(X_test.shape)

#clf = GradientBoostingClassifier(n_estimators=3000, learning_rate=0.009, random_state=5) #0.85897 score on kaggle
clf = RandomForestClassifier(n_estimators=3000, max_features='sqrt', random_state=5)  #0.85977 score on kaggle

#print(np.ravel(y_train))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_validation)
temp = 0
for i in range(len(y_pred)):
    if y_pred[i] == y_validation[i]:
        temp+=1
print(temp/len(y_pred))
y_test = clf.predict(X_test)
submission = pd.DataFrame({'Id': test_uniq_id, "Fault": y_test})
submission.to_csv('submission_16.csv', index=False)
