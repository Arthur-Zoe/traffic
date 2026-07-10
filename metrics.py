import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix


def compute_metrics(y_true, y_pred, labels=None):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)),
    }


def save_report(y_true, y_pred, classes, out_txt, out_csv=None):
    text = classification_report(y_true, y_pred, target_names=classes, digits=6, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(text)
        f.write('\nConfusion matrix, rows=true, cols=pred\n')
        f.write(np.array2string(cm))
    if out_csv is not None:
        import pandas as pd
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(out_csv, encoding='utf-8-sig')
