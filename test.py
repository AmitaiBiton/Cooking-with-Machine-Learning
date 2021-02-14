from finalProject import FrequencyCalculationPerCountryPerItem, \
    get3Classes, \
    MaxOnTheta, Logistic_Regression, \
     AdaBoost
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder





if  __name__ == "__main__" :
    dataTrain = pd.read_json('train.json')
    dataTrain.head()

    #counters = FrequencyCalculationPerCountryPerItem(dataTrain)

    train3Classes = get3Classes(dataTrain)

    # Number of dishes for country
    #plt.style.use('ggplot')
    #dataTrain['cuisine'].value_counts().plot(kind='bar')
    #plt.show()
    # All data
    dataTrain['all_ingredients'] =dataTrain['ingredients'].map(",".join )
    dataTrain.head()

    # Data only for 3 most common cuisine
    train3Classes['all_ingredients'] = train3Classes['ingredients'].map(",".join)
    train3Classes.head()

    cv = CountVectorizer()
    X3Classes = cv.fit_transform(train3Classes['all_ingredients'].values)
    X = cv.fit_transform(dataTrain['all_ingredients'].values)
    enc = LabelEncoder()
    y3Classes = enc.fit_transform(train3Classes.cuisine)
    y = enc.fit_transform(dataTrain.cuisine)

    features300= MaxOnTheta(X3Classes , y3Classes , num=300)

    features100 = MaxOnTheta(X3Classes, y3Classes, num=100)

    # all features and classes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # all features but 6 classes
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X3Classes, y3Classes, test_size=0.2, random_state=42)

    # 300 predict by 300 features
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X3Classes[: ,features300], y3Classes, test_size=0.2, random_state=42)

    # show the dif  uf we take  100 feature or  300
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X3Classes[: ,features100], y3Classes, test_size=0.2, random_state=42)


    AdaBoost(X_train, X_test, y_train, y_test ,dataTrain)
    AdaBoost(X_train1, X_test1, y_train1, y_test1 ,train3Classes)
    AdaBoost(X_train2, X_test2, y_train2, y_test2 ,train3Classes)
    AdaBoost(X_train3, X_test3, y_train3, y_test3 ,train3Classes)

    Logistic_Regression(X_train, X_test, y_train, y_test , dataTrain)
    Logistic_Regression(X_train1, X_test1, y_train1, y_test1 , train3Classes)
    Logistic_Regression(X_train2, X_test2, y_train2, y_test2 , train3Classes)
    Logistic_Regression(X_train3, X_test3, y_train3, y_test3 , train3Classes)

    """
    clf = SVC(decision_function_shape='ovr')
    model = OneVsOneClassifier(clf)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    """


