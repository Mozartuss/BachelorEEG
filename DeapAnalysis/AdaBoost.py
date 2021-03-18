import multiprocessing as mp
from os.path import sep

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier


def normal_abc(dataset, n, rate):
    x_train, x_test, y_train, y_test = dataset
    abc_normal = AdaBoostClassifier(n_estimators=n, learning_rate=rate)
    model_normal = abc_normal.fit(x_train, y_train)
    y_pred_normal = model_normal.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred_normal)
    print("AdaBoost with\t {}\t iterations and\t {}\t learning rate -> Accuracy:".format(n, rate), np.round(acc, 2))
    return [n, rate, acc]


def normal_xgbc(dataset):
    x_train, x_test, y_train, y_test = dataset
    xgbc = XGBClassifier(use_label_encoder=False)
    xgbc.fit(x_train, y_train)
    y_pred_xgbc = xgbc.predict(x_test)
    print("XGBoost -> Accuracy:", np.round(metrics.accuracy_score(y_test, y_pred_xgbc), 2))


def advanced_xgbr(dataset, depth, lr):
    x_train, x_test, y_train, y_test = dataset
    xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=lr, max_depth=depth,
                              n_estimators=10)
    xg_reg.fit(x_train, y_train)
    y_pred_xgbr = xg_reg.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred_xgbr)
    print("XGBoost advanced with depth\t {}\t and\t {}\t learning rate -> Accuracy:".format(depth, lr),
          np.round(acc, 2))
    return [depth, lr, acc]


def svc_abc(dataset, kernel, decision):
    x_train, x_test, y_train, y_test = dataset
    abc_svc = AdaBoostClassifier(n_estimators=5000,
                                 base_estimator=SVC(probability=True, kernel=kernel, decision_function_shape=decision),
                                 learning_rate=0.01)
    model_svc = abc_svc.fit(x_train, y_train)
    y_pred_svc = model_svc.predict(x_test)
    print("AdaBoost svc with kernel\t {}\t and\t {}\t decision -> Accuracy:".format(kernel, decision),
          np.round(metrics.accuracy_score(y_test, y_pred_svc), 2))


def load_test_data(data_path, label_path, bins, type):
    # Load data
    features = pd.read_csv(data_path)
    labels = pd.read_csv(label_path)
    X = features.values[:, 1:]
    y = pd.DataFrame(np.digitize(labels[type].values, bins, right=True)).values[:, 0]
    return X, y


if __name__ == '__main__':
    path = ".." + sep + ".." + sep + ".." + sep + "DEAP Dataset" + sep + "Preprocessed Datasets" + sep + "data_preprocessed_python" + sep + "Participants" + sep + "CompleteFeatures" + sep
    data = "SelectedFeaturesArousal.csv"
    label = "CompleteFeaturesLabels.csv"
    # bins = -2=[0-1] -1=[1-2.25] 0=[2.25-5] 1=[5-7.25] 2=[7.25-9.9999] -> scale Label from -2 to 2
    X, y = load_test_data(path + data, path + label, np.array([1, 2.25, 5, 7.25, 9]), "arousal")

    rates = np.arange(0.01, 1.00, 0.01)
    iters = np.arange(500, 100000, 250)
    dt = train_test_split(X, y, test_size=0.3)

    pool = mp.Pool(mp.cpu_count() - 1)
    result_abc = pool.starmap(normal_abc,
                              [(dt, n, np.round(rate, 2)) for n in iters for rate in
                               rates])
    result_abc = sorted(result_abc, key=lambda x: x[-1], reverse=True)
    print(result_abc[0])
    pool.apply(normal_xgbc, args=dt)

    result_xgbr = pool.starmap(advanced_xgbr,
                               [(dt, depth, np.round(rate, 2)) for depth in np.arange(5, 1280, 5) for rate in
                                rates])
    result_xgbr = sorted(result_xgbr, key=lambda x: x[-1], reverse=True)
    print(result_xgbr[0])
    pool.starmap(svc_abc,
                 [(dt, k, d) for k in ["linear", "rbf", "poly", "sigmoid"] for d in
                  ["ovo", "ovr"]])
    pool.close()
