"""
DS-36 — Kalp Hastalığı Riski (Random Forest + Feature Importance)

Bir hastanede **klinik karar destek sistemi** kuruyorsun. Hastanın yaş,
tansiyon, kolesterol, göğüs ağrısı tipi gibi ölçümlerinden kalp hastalığı
riskini tahmin edeceksin. Random Forest hem güçlü bir model, hem de
**feature_importances_** ile "hangi faktör en kritik?" sorusunu yanıtlayarak
doktora business insight veriyor.

Her fonksiyonun pass kısmını doldur. Testleri çalıştır, hepsi geçene kadar
iterate et: `python watch.py` veya `pytest tests/test_question.py -v`

NOT: Random Forest mesafe tabanlı DEĞİL (ağaç tabanlı) → **scaling GEREKMEZ**.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Veri setini yükle
def load_heart_data(path='data/heart.csv'):
    """
    data/heart.csv dosyasını oku, feature'ları (X) ve hedefi (y) ayır.

    Args:
        path: csv dosyasının yolu (default: 'data/heart.csv')

    Returns:
        tuple: (X_df, y)
            X_df: pd.DataFrame — 13 feature sütunu
                  (age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                   exang, oldpeak, slope, ca, thal)
            y: pd.Series — 'target' sütunu (1: kalp hastalığı var, 0: yok)

    İpucu:
    - df = pd.read_csv(path)
    - y = df['target']
    - X = df.drop(columns=['target'])
    """
    df = pd.read_csv(path)
    y = df['target']
    X = df.drop(columns=['target'])
    return X, y


# 2. Veriyi keşfet
def explore_data(X, y):
    """
    Veri seti hakkında özet bilgi üret.

    Args:
        X: feature DataFrame
        y: hedef Series

    Returns:
        dict: {
            'n_samples': int (örnek/hasta sayısı),
            'n_features': int (feature sayısı = 13),
            'positive_count': int (kalp hastalığı OLAN hasta sayısı, y==1),
            'negative_count': int (kalp hastalığı OLMAYAN hasta sayısı, y==0),
            'feature_names': list (feature isimleri),
        }

    İpucu:
    - len(X) → örnek sayısı
    - X.shape[1] → feature sayısı
    - (y == 1).sum() → pozitif sayısı
    - list(X.columns) → feature isimleri
    """
    return {
        'n_samples': int(len(X)),
        'n_features': int(X.shape[1]),
        'positive_count': int((y == 1).sum()),
        'negative_count': int((y == 0).sum()),
        'feature_names': list(X.columns),
    }


# 3. Train / test böl
def split_data(X, y):
    """
    X ve y'yi train/test olarak böl.

    - test_size = 0.2
    - stratify = y (sınıf oranı korunsun)
    - random_state = 42

    Args:
        X: feature DataFrame
        y: hedef Series

    Returns:
        tuple: (X_train, X_test, y_train, y_test)

    İpucu:
    - from sklearn.model_selection import train_test_split
    """
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# 4. Random Forest eğit
def train_random_forest(X_train, y_train, n_estimators=100):
    """
    RandomForestClassifier kur ve eğit.

    Args:
        X_train, y_train: eğitim verileri
        n_estimators: ağaç sayısı (default 100)

    Returns:
        sklearn RandomForestClassifier: eğitilmiş model
            (n_estimators=n_estimators, random_state=42)

    İpucu:
    - from sklearn.ensemble import RandomForestClassifier
    - model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    - model.fit(X_train, y_train)
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model


# 5. Modeli değerlendir
def evaluate_model(model, X_test, y_test):
    """
    Test seti üzerinde tahmin yap ve metrikleri hesapla.

    Returns:
        dict: {
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1': float,
            'confusion_matrix': np.array (shape (2, 2)),
        }

    İpucu:
    - from sklearn.metrics import accuracy_score, precision_score,
      recall_score, f1_score, confusion_matrix
    - y_pred = model.predict(X_test)
    - precision/recall/f1 için zero_division=0 kullanabilirsin.
    """
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }


# 6. Feature importance — hangi faktör kalp hastalığını belirliyor?
def get_feature_importances(model, feature_names, top_n=10):
    """
    Modelin feature_importances_ değerlerini feature isimleriyle eşleştir,
    önem sırasına göre (büyükten küçüğe) sırala, en önemli top_n tanesini dön.

    Args:
        model: eğitilmiş RandomForestClassifier
        feature_names: feature isimleri listesi
        top_n: kaç tane dönsün (default 10)

    Returns:
        list of tuple: [(feature_name, importance), ...] — önem azalan sırada,
                       en fazla top_n eleman.

    İpucu:
    - importances = model.feature_importances_
    - pairs = list(zip(feature_names, importances))
    - pairs.sort(key=lambda p: p[1], reverse=True)
    - pairs[:top_n]
    """
    importances = model.feature_importances_
    pairs = list(zip(feature_names, importances))
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs[:top_n]


# 7. Feature importance barplot
def plot_feature_importances(model, feature_names):
    """
    Feature importance'ları yatay bar grafik (barh) olarak çiz.

    Args:
        model: eğitilmiş RandomForestClassifier
        feature_names: feature isimleri

    Returns:
        matplotlib Figure objesi

    İpucu:
    - import matplotlib.pyplot as plt
    - get_feature_importances ile sıralı listeyi al
    - fig, ax = plt.subplots()
    - ax.barh(names, values)  # en önemli en üstte görünsün diye listeyi ters çevir
    - return fig
    """
    pairs = get_feature_importances(model, feature_names, top_n=len(feature_names))
    names = [p[0] for p in pairs][::-1]
    vals = [p[1] for p in pairs][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, vals, color="#c0392b")
    ax.set_xlabel("Önem (feature importance)")
    ax.set_title("Random Forest — Kalp Hastalığını Belirleyen Faktörler")
    fig.tight_layout()
    return fig


# 8. n_estimators tuning
def tune_n_estimators(X_train, X_test, y_train, y_test, n_list):
    """
    Verilen her n_estimators değeri için bir Random Forest eğit, test
    accuracy'sini ölç.

    Args:
        X_train, X_test, y_train, y_test: bölünmüş veri
        n_list: denenecek n_estimators değerleri listesi (örn [10, 50, 100, 200])

    Returns:
        dict: {n: test_accuracy, ...} her n_estimators için accuracy

    İpucu:
    - Her n için train_random_forest(..., n_estimators=n) eğit
    - evaluate_model(...)['accuracy'] ile ölç
    """
    results = {}
    for n in n_list:
        model = train_random_forest(X_train, y_train, n_estimators=n)
        results[n] = evaluate_model(model, X_test, y_test)['accuracy']
    return results


# 9. Decision Tree vs Random Forest
def compare_dt_vs_rf(X_train, X_test, y_train, y_test):
    """
    Tek bir Decision Tree ile Random Forest'ı aynı veride karşılaştır.
    Random Forest genellikle daha kararlı/yüksek skor verir (ensemble gücü).

    Returns:
        dict: {
            'decision_tree': float (DT test accuracy),
            'random_forest': float (RF test accuracy),
        }

    İpucu:
    - from sklearn.tree import DecisionTreeClassifier
    - DecisionTreeClassifier(random_state=42).fit(...)
    - RF için train_random_forest(..., n_estimators=100) kullan
    - from sklearn.metrics import accuracy_score
    """
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt.predict(X_test))
    rf = train_random_forest(X_train, y_train, n_estimators=100)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    return {'decision_tree': dt_acc, 'random_forest': rf_acc}


# 10. OOB (Out-Of-Bag) skoru
def get_oob_score(X_train, y_train, n_estimators=100):
    """
    OOB skoru = her ağacın eğitiminde KULLANILMAYAN (bootstrap dışı kalan)
    örnekler üzerinde ölçülen accuracy. Ücretsiz bir validation skoru gibidir
    — ayrı validation seti ayırmadan modelin genelleme gücünü tahmin eder.

    Args:
        X_train, y_train: eğitim verileri
        n_estimators: ağaç sayısı (default 100)

    Returns:
        float: model.oob_score_

    İpucu:
    - RandomForestClassifier(n_estimators=n_estimators, oob_score=True,
      random_state=42)
    - .fit(X_train, y_train) sonrası .oob_score_ oku
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators, oob_score=True, random_state=42,
    )
    model.fit(X_train, y_train)
    return float(model.oob_score_)


# 11. Tek hasta için tahmin
def predict_patient(model, patient_row):
    """
    Tek bir hasta için kalp hastalığı tahmini yap.

    Args:
        model: eğitilmiş RandomForestClassifier
        patient_row: tek hasta verisi (dict {feature: value} VEYA 1D list/array,
                     13 feature sıralı)

    Returns:
        dict: {
            'prediction': 0 veya 1 (int),
            'risk': 'yüksek' (prediction==1) / 'düşük' (prediction==0),
            'probability': float (kalp hastalığı olasılığı, pozitif sınıf),
        }

    İpucu:
    - dict ise → pd.DataFrame([patient_row])
    - list/array ise → np.array(patient_row).reshape(1, -1)
    - pred = int(model.predict(X_new)[0])
    - proba = float(model.predict_proba(X_new)[0, 1])
    """
    if isinstance(patient_row, dict):
        X_new = pd.DataFrame([patient_row])
    else:
        X_new = np.array(patient_row).reshape(1, -1)
    pred = int(model.predict(X_new)[0])
    proba = float(model.predict_proba(X_new)[0, 1])
    return {
        'prediction': pred,
        'risk': 'yüksek' if pred == 1 else 'düşük',
        'probability': proba,
    }


# 12. Cross-validation
def cross_validate_rf(X, y, cv=5):
    """
    Random Forest'ı cross-validation ile değerlendir (accuracy).

    Args:
        X, y: tüm veri
        cv: fold sayısı (default 5)

    Returns:
        dict: {
            'mean': float (ortalama CV accuracy),
            'std': float (standart sapma),
        }

    İpucu:
    - from sklearn.model_selection import cross_val_score
    - model = RandomForestClassifier(n_estimators=100, random_state=42)
    - scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    - scores.mean(), scores.std()
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return {'mean': float(scores.mean()), 'std': float(scores.std())}


# 13. Sade model — sadece en önemli feature'larla
def simplify_with_top_features(X_train, X_test, y_train, y_test, feature_names, top_n=5):
    """
    Önce tüm feature'larla bir RF eğit ve accuracy ölç. Sonra sadece en
    önemli top_n feature ile yeni bir RF eğit ve accuracy ölç. Sade model
    neredeyse aynı performansı veriyor mu? (klinikte daha az ölçüm = daha ucuz)

    Args:
        X_train, X_test, y_train, y_test: bölünmüş veri
        feature_names: feature isimleri
        top_n: kaç feature kullanılsın (default 5)

    Returns:
        dict: {
            'all_features_acc': float (tüm feature'larla),
            'top5_acc': float (sadece en önemli top_n feature ile),
        }

    İpucu:
    - Tüm feature'larla RF eğit, get_feature_importances ile en iyi top_n
      feature isimlerini al
    - X_train[top_feature_names] ile alt küme seç (DataFrame sütun seçimi)
    - Alt küme ile yeni RF eğit, evaluate_model ile ölç
    """
    full_model = train_random_forest(X_train, y_train, n_estimators=100)
    all_acc = evaluate_model(full_model, X_test, y_test)['accuracy']

    top = [name for name, _ in get_feature_importances(full_model, feature_names, top_n=top_n)]
    X_train_top = X_train[top]
    X_test_top = X_test[top]
    top_model = train_random_forest(X_train_top, y_train, n_estimators=100)
    top_acc = evaluate_model(top_model, X_test_top, y_test)['accuracy']

    return {'all_features_acc': all_acc, 'top5_acc': top_acc}


# 14. Tüm pipeline'ı uçtan uca çalıştır
def run_pipeline():
    """
    Yukarıdaki fonksiyonları birleştirip tam akışı çalıştır:
    1. Veri yükle (X, y)
    2. Train/test böl
    3. Random Forest eğit (n_estimators=100)
    4. Test accuracy ölç
    5. En önemli feature'ı bul (get_feature_importances top_n=1)
    6. OOB skoru hesapla
    7. DT vs RF karşılaştır, farkı hesapla (rf_acc - dt_acc)

    Returns:
        dict: {
            'n_samples': int,
            'test_accuracy': float,
            'top_feature': str (en kritik feature ismi),
            'oob_score': float,
            'dt_vs_rf_diff': float (random_forest - decision_tree accuracy),
        }
    """
    X, y = load_heart_data()
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_random_forest(X_train, y_train, n_estimators=100)
    test_accuracy = evaluate_model(model, X_test, y_test)['accuracy']

    top_feature = get_feature_importances(model, feature_names, top_n=1)[0][0]
    oob_score = get_oob_score(X_train, y_train, n_estimators=100)

    comp = compare_dt_vs_rf(X_train, X_test, y_train, y_test)
    dt_vs_rf_diff = comp['random_forest'] - comp['decision_tree']

    return {
        'n_samples': int(len(X)),
        'test_accuracy': test_accuracy,
        'top_feature': top_feature,
        'oob_score': oob_score,
        'dt_vs_rf_diff': dt_vs_rf_diff,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print("📊 Pipeline Sonuçları:")
    print(f"  Örnek sayısı       : {result['n_samples']}")
    print(f"  Test accuracy      : {result['test_accuracy']:.2%}")
    print(f"  En kritik faktör   : {result['top_feature']}")
    print(f"  OOB score          : {result['oob_score']:.2%}")
    print(f"  RF - DT farkı      : {result['dt_vs_rf_diff']:+.2%}")
