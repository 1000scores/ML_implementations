import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression


class MyLogisticRegression():
    def __init__(self):
        self.W = None
        self.b = None
        self.loss = []
        self.iters = 0
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __predict_proba(self, X, W, b):
        return W.dot(X.T) + b
    
    def predict_proba(self, X):
        prediction = self.__predict_proba(X, self.W, self.b)
        return np.array([(1 - elem, elem) for elem in prediction])

    def __predict(self, X, W, b, thresh):
        pred = lambda x: x >= thresh
        return pred(W.dot(X.T) + b).astype(int)
    
    def predict(self, X, thresh=0.5):
        return self.__predict(X, self.W, self.b, thresh)

    def cost_f(self, X, y_true, y_pred):
        return np.mean(-y_true * np.log(self.sigmoid(y_pred)) - (1 - y_true) * np.log(1 - self.sigmoid(y_pred)))

    def dw_df(self, X, y_true, y_pred, m_samples):
        return (1 / m_samples) * (X.T.dot(y_pred - y_true))

    def db_df(self, X, y_true, y_pred):
        return np.mean((y_pred - y_true))

    def update_W_b(self, W, b, dw_df, db_df, lr):
        return (W - lr * dw_df, b - lr * db_df)
    
    def plot_loss(self, path="loss_plot.png"):
        plt.plot(list(range(self.iters)), self.loss)
        plt.show()
        plt.savefig(path)
    
    def fit(self, X, y, iters=100, lr=0.01):
        assert X.shape[0] == y.shape[0]
        self.iters = iters
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
        
if __name__ == '__main__':
    X, y = make_classification(n_features=10, n_samples=1000, random_state=78)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=78)
    my_clf = MyLogisticRegression().fit(X_train, y_train)
    my_clf.plot_loss()
    test_clf(
        my_clf,
        X_test,
        title="My regression"
    )
    test_clf(
        LogisticRegression(random_state=78).fit(X_train, y_train),
        X_test,
        title="Sklearn regression"
    )
