import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class MyLogisticRegression():
    def __init__(self):
        self.W = None
        self.b = None
        self.loss = []
        self.iters = 0
        self.l1_weight = 0 # Lasso
        self.l2_weight = 0 # Ridge
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __predict_proba(self, X, W, b):
        return self.sigmoid(W.dot(X.T) + b)
    
    def predict_proba(self, X):
        prediction = self.__predict_proba(X, self.W, self.b)
        return np.array([(1 - elem, elem) for elem in prediction])

    def __predict(self, X, W, b, thresh):
        pred = lambda x: x >= thresh
        return self.__predict_proba(X, W, b).astype(int)
    
    def predict(self, X, thresh=0.5):
        return self.__predict(X, self.W, self.b, thresh)
        
    def cost_f(self, X, y_true, y_pred):
        return np.mean(
            -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
        ) + \
        self.l1_weight * np.sum(np.sign(self.W)) +\
        self.l2_weight * np.sum(self.W ** 2)

    def dw_df(self, X, y_true, y_pred, m_samples):
        return (1 / m_samples) * (
            (X.T.dot(y_pred - y_true)) +\
            self.l1_weight * np.sign(self.W) +\
            2 * self.l2_weight * self.W
        )

    def db_df(self, X, y_true, y_pred):
        return np.mean((y_pred - y_true))

    def update_W_b(self, W, b, dw_df, db_df, lr):
        return (W - lr * dw_df, b - lr * db_df)
    
    def plot_loss(self, label):
        plt.plot(list(range(self.iters)), self.loss, label=label)
        plt.legend(loc='upper center')
        plt.show()
    
    def fit(
        self,
        X,
        y,
        iters=100,
        lr=0.01,
        l1_weight=0,
        l2_weight=0
    ):
        assert X.shape[0] == y.shape[0]
        self.iters = iters
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        m_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        self.b = 0
        for i in range(iters):
            y_pred = self.__predict_proba(X, self.W, self.b)
            self.loss.append(self.cost_f(X, y, y_pred))
            dw = self.dw_df(X, y, y_pred, m_samples)
            db = self.db_df(X, y, y_pred)
            self.W, self.b = self.update_W_b(self.W, self.b, dw, db, lr)
            
        return self
        

def test_clf(clf, X_test, title="Unknown"):
    print(title)
    print(f"Accuracy: {accuracy_score(y_test, clf.predict(X_test))}")
    print(f"F1: {f1_score(y_test, clf.predict(X_test))}")
    print(f"roc_auc: {roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])}")
    
    
def framingham_dataset():
    df = pd.read_csv("/home/denis/Documents/ML/implementations/logistic_regression/framingham.csv").dropna()
    #print(df.isnull().sum())
    #print(df)
    #print(df.describe())
    y = df["TenYearCHD"].to_numpy()
    X = df.drop(["TenYearCHD"], axis=1)#.to_numpy()
    #print(X)
    #print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=78)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def sklearn_datset():
    X, y = make_classification(n_features=10, n_informative=5, n_redundant=5, n_samples=1000, random_state=34)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=78)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = sklearn_datset()
    my_clf = MyLogisticRegression().fit(X_train, y_train)
    print(my_clf.W)
    print(my_clf.b)
    my_clf_l1 = MyLogisticRegression().fit(X_train, y_train, l1_weight=1)
    print(my_clf_l1.W)
    print(my_clf_l1.b)
    my_clf_l2 = MyLogisticRegression().fit(X_train, y_train, l2_weight=1)
    print(my_clf_l2.W)
    print(my_clf_l2.b)
    my_clf_l1_l2 = MyLogisticRegression().fit(X_train, y_train, l1_weight=1, l2_weight=1)
    print(my_clf_l1_l2.W)
    print(my_clf_l1_l2.b)
    my_clf.plot_loss(label="Base")
    my_clf_l1.plot_loss(label="L1")
    my_clf_l2.plot_loss(label="L2")
    my_clf_l1_l2.plot_loss(label="L1+L2")
    plt.savefig("loss.png")
    test_clf(
        my_clf,
        X_test,
        title="My regression"
    )
    test_clf(
        my_clf_l1,
        X_test,
        title="My regression with L1"
    )
    test_clf(
        my_clf_l2,
        X_test,
        title="My regression with L2"
    )
    test_clf(
        my_clf_l1_l2,
        X_test,
        title="My regression with L1 + L2"
    )
    sk = LogisticRegression(penalty="l1", solver="liblinear", C=1, random_state=78).fit(X_train, y_train)
    test_clf(
        sk,
        X_test,
        title="Sklearn regression with L1"
    )
    print(sk.coef_)