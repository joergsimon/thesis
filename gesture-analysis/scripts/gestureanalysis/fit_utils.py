import gc
import tqdm
from sklearn.metrics import f1_score
import sklearn.metrics as metr
from sklearn.base import clone


def fit_classifier(classifiers, X, y, Xval, yval):
    clf = None
    clf_n = None

    results = []
    progress = tqdm.tqdm_notebook(classifiers)
    for clf_name, clf_candidate in progress:
        progress.set_description(clf_name)
        clf_candidate = clone(clf_candidate)
        gc.collect()
        progress.set_description(clf_name + ' cloned')
        clf_candidate.fit(X, y)
        progress.set_description(clf_name + ' fittet')
        p = clf_candidate.predict(Xval)
        s = f1_score(yval, p, average='micro')
        p_t = clf_candidate.predict(X)
        s_t = f1_score(y, p_t, average='micro')
        print(f"val score {s} for {clf_name} (test: {s_t})")
        cmatrix = metr.confusion_matrix(yval, p)
        # print(cmatrix)
        results.append((clf_name, s, s_t, cmatrix, clf_candidate))
        gc.collect()

    return results
