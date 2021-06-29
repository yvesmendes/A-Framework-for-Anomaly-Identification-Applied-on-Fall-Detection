from sklearn.decomposition import TruncatedSVD, PCA
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import cohen_kappa_score, roc_curve, auc, confusion_matrix, accuracy_score
import sklearn
from sklearn.preprocessing import StandardScaler, Normalizer
from keras.layers import Input, Dense
import keras.regularizers
from keras import Model, Sequential
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
import cv2
from tensorflow import keras

mse_thr = 513  # PRECIS-SVD

# PREP Parameters
n_classes = 2
split_per = 0.9
length_dataset = 256  # precis_500
db = 'PRECIS_HAR'
# db = 'UrFall'
db = 'UpFall'
# model = 'AE' #FOREST, SVD, PCA, SVM, AE
model = 'SVD'  # FOREST, SVD, PCA, SVM, AE
number_of_experiments = 2
use_dynamic_threshold = True
n_components = 47
# AE Parameter
nb_epoch = 100
encoding_dim = 1024
batch_size = 64
batch_size = 16


def prep2():
    adl_train = np.zeros((1, length_dataset))
    adl_test = np.zeros((1, length_dataset))
    fall = np.zeros((1, length_dataset))

    fullBase = np.load('./features/' + db + '/' + str(length_dataset) + 'full.npy')

    class_label = fullBase[:, -1]

    maxLabel = class_label.max(axis=0)

    fall = fullBase[fullBase[:, -1] == maxLabel]
    fall = fall[:, 0:length_dataset]
    for x in range(0, int(maxLabel)):
        adl_temp = fullBase[fullBase[:, -1] == x]
        adl_temp = adl_temp[:, 0:length_dataset]
        np.random.shuffle(adl_temp)
        train = adl_temp[:int(adl_temp.shape[0] * split_per)]
        test = adl_temp[int(adl_temp.shape[0] * split_per):]
        adl_train = np.concatenate([adl_train, train], axis=0)
        adl_test = np.concatenate([adl_test, test], axis=0)

    adl_train = np.delete(adl_train, (0), axis=0)
    adl_test = np.delete(adl_test, (0), axis=0)

    adl_train_lbl = np.zeros((adl_train.shape[0], 1))

    for x in range(adl_train.shape[0]):
        adl_train_lbl[x] = 0

    adl_test_lbl = np.zeros((adl_test.shape[0], 1))
    for x in range(adl_test.shape[0]):
        adl_test_lbl[x] = 0

    fall_lbl = np.zeros((fall.shape[0], 1))
    for x in range(fall.shape[0]):
        fall_lbl[x] = 1

    adl_train_lbl = np_utils.to_categorical(adl_train_lbl, n_classes)
    adl_test_lbl = np_utils.to_categorical(adl_test_lbl, n_classes)

    fall_lbl = np_utils.to_categorical(fall_lbl, n_classes)

    print('adl_train', adl_train.shape)
    print('adl_train_lbl', adl_train_lbl.shape)
    print('adl_test', adl_test.shape)
    print('adl_test_lbl', adl_test_lbl.shape)
    print('fall', fall.shape)

    return adl_train, adl_train_lbl, adl_test, adl_test_lbl, fall, fall_lbl


def classify(Xs, threshold):
    pred = (Xs >= threshold) * 1
    return pred


def get_metrics(y, Y_pred, threshold=None, cal_roc=False, complete=True):
    y_true = None
    y_pred = None
    if model == 'AE' or model == 'VAE':
        y_true = y
        y_pred = Y_pred
    else:
        if threshold is None:
            y_pred = np.argmax(Y_pred, axis=-1)
        else:
            y_pred = (Y_pred[:, 1] >= threshold) * 1

        y_true = np.argmax(y, axis=-1)
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)

    if complete:
        TP = float(matrix[0][0])
        FP = float(matrix[0][1])
        FN = float(matrix[1][0])
        TN = float(matrix[1][1])

        if (TP + FN) > 0:
            Sensitivity = TP / (TP + FN)
        else:
            Sensitivity = 0
        if (TN + FP) > 0:
            Specificity = TN / (TN + FP)
        else:
            Specificity = 0

        if (TP + FP) > 0:
            Precision = TP / (TP + FP)
        else:
            Precision = (TP + FP)

        Recall = Sensitivity

        if (Recall > 0) and (Precision > 0):
            F1 = 2 / ((Recall ** -1) + (Precision ** -1))
        else:
            F1 = 0

    metrics = {}
    metrics["acc"] = acc
    metrics["kappa"] = kappa
    metrics["matrix"] = matrix
    if complete:
        metrics["Sensitivity"] = Sensitivity
        metrics["Specificity"] = Specificity
        metrics["F1"] = F1
        metrics["Precision"] = Precision

    if cal_roc:
        y_roc = y_pred
        if model != 'AE' and model != 'VAE':
            y_roc = Y_pred[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true, y_roc, pos_label=1)
        valor_auc = auc(fpr, tpr)
        metrics["AUC"] = valor_auc
        metrics["thresholds"] = thresholds
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.4f) ' % (valor_auc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig(dataset + 'output_roc')
        plt.close()

    return metrics


def run_svd(svd, data, lbl, name, mse_thr):
    trans = svd.transform(data)
    inv_trans = svd.inverse_transform(trans)

    print(name, sklearn.metrics.mean_squared_error(data, inv_trans))

    mse = np.mean(np.power(data - inv_trans, 2), axis=1)
    pred = None
    pred = classify(mse, mse_thr)

    pred = np_utils.to_categorical(pred, n_classes)
    metrics = get_metrics(lbl, pred, complete=False)

    return pred


def get_model_precis_har(encoding_dim, hidden_dim, learning_rate):
    input_dim = 256
    activation = 'relu'
    model = Sequential()
    model.add(Dense(1024, input_shape=(256,), activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(256, activation=activation))
    model.compile(metrics=['accuracy'],
                  loss='mean_squared_error',
                  optimizer='adam')

    model.summary()
    return model


def get_model_ur_fall():
    input_dim = 256
    activation = 'relu'
    model = Sequential()
    model.add(Dense(256, input_shape=(256,), activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(512, activation=activation))
    model.add(Dense(512, activation=activation))
    model.add(Dense(512, activation=activation))
    model.add(Dense(512, activation=activation))
    model.add(Dense(256, activation=activation))
    model.compile(metrics=['accuracy'],
                  loss='mean_squared_error',
                  optimizer='adam')

    model.summary()
    return model


def get_model_up_fall(encoding_dim, hidden_dim, learning_rate):
    input_dim = 256
    activation = 'linear'
    model = Sequential()
    model.add(Dense(256, input_shape=(256,), activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(1024, activation=activation))
    model.add(Dense(256, activation=activation))
    model.compile(metrics=['accuracy'],
                  loss='mean_squared_error',
                  optimizer='adam')

    model.summary()
    return model


def get_model_b(encoding_dim, hidden_dim, learning_rate):
    input_dim = 256
    activation = 'linear'
    model = Sequential()
    model.add(Dense(1024, input_shape=(256,), activation=activation))
    model.add(Dense(64, activation=activation, activity_regularizer=keras.regularizers.l1(1e-3)))
    model.add(Dense(32, activation=activation, activity_regularizer=keras.regularizers.l1(1e-3)))
    model.add(Dense(256, activation=activation))
    model.add(Dense(256, activation=activation))
    model.compile(metrics=['accuracy'],
                  loss='mean_squared_error',
                  optimizer='adam')

    model.summary()
    return model


def get_model_old_new(encoding_dim, hidden_dim, learning_rate):
    input_dim = 256
    activation = 'linear'
    model = Sequential()
    model.add(Dense(512, input_shape=(256,), activation=activation))
    model.add(Dense(1024, activation=activation, activity_regularizer=keras.regularizers.l1(1e-3)))
    model.add(Dense(16, activation=activation, activity_regularizer=keras.regularizers.l1(1e-3)))
    model.add(Dense(256, activation=activation))
    model.add(Dense(256, activation=activation))
    model.compile(metrics=['accuracy'],
                  loss='mean_squared_error',
                  optimizer='adam')

    model.summary()
    return model


def get_model_old(encoding_dim, hidden_dim, learning_rate):
    input_dim = 256
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="relu",
                    activity_regularizer=keras.regularizers.l1(learning_rate))(input_layer)
    encoder = Dense(hidden_dim, activation="relu")(encoder)
    decoder = Dense(hidden_dim, activation="relu")(encoder)
    decoder = Dense(encoding_dim, activation="relu")(decoder)
    decoder = Dense(input_dim, activation="linear")(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(metrics=['accuracy'],
                        loss='mean_squared_error',
                        optimizer='adam')

    return autoencoder


def prep_ae(db):
    fullBase = np.load('./features/' + db + '/' + str(length_dataset) + 'full.npy')
    path_save = "results-single/"
    dbName = db
    # db = sigmoid(np.load(path_db))
    db = fullBase

    label_fall = np.amax(db[:, -1])
    adls = db[np.where(db[:, -1] != label_fall)]
    falls = db[np.where(db[:, -1] == label_fall)]

    y_adls = adls[:, -1]
    adls = adls[:, :-1]

    y_falls = falls[:, -1]
    falls = falls[:, :-1]

    print("db")
    print(db.shape)
    print("adls")
    print(adls.shape)
    print("falls")
    print(falls.shape)

    rand = random.randint(0, 100)

    train_adl, val_adl, y_train_adl, y_val_adl = train_test_split(adls, y_adls, stratify=y_adls,
                                                                  test_size=1 - split_per,
                                                                  random_state=rand, shuffle=True)

    print("train_adl", "y_train_adl")

    if dbName != 'UrFall':
        train_adl = cv2.normalize(train_adl, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        train_adl = ((train_adl.astype(np.float32) - 127.5) / 127.5)

    scaler = None

    if dbName == 'UrFall':
        scaler = Normalizer()
    else:
        scaler = StandardScaler()

    scaler.fit(train_adl)
    train_adl = scaler.transform(train_adl)
    print(train_adl.shape, y_train_adl.shape)
    y_train_adl = np.zeros(y_train_adl.shape)

    teste_adl = val_adl[math.ceil(val_adl.shape[0] * 0.5):]
    y_teste_adl = np.zeros(teste_adl.shape[0])

    if dbName != 'UrFall':
        teste_adl = cv2.normalize(teste_adl, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        teste_adl = ((teste_adl.astype(np.float32) - 127.5) / 127.5)

    print("teste_adl", "y_teste_adl")
    teste_adl = scaler.transform(teste_adl)
    print(teste_adl.shape, y_teste_adl.shape)

    y_val_adl = np.zeros(math.ceil(val_adl.shape[0] * 0.5))
    val_adl = val_adl[:math.ceil(val_adl.shape[0] * 0.5)]

    print("val_adl", "y_val_adl")
    print(val_adl.shape, y_val_adl.shape)

    if dbName != 'UrFall':
        val_adl = cv2.normalize(val_adl, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        val_adl = ((val_adl.astype(np.float32) - 127.5) / 127.5)

    val_adl = scaler.transform(val_adl)

    if dbName != 'UrFall':
        falls = cv2.normalize(falls, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        falls = ((falls.astype(np.float32) - 127.5) / 127.5)

    falls = scaler.transform(falls)

    y_falls = np.ones(y_falls.shape)
    print("X_falls", "y_falls")
    print(falls.shape, y_falls.shape)

    X_teste = np.concatenate((falls, teste_adl))
    y_teste = np.concatenate((y_falls, y_teste_adl))

    print("X_teste", "y_teste")
    print(X_teste.shape, y_teste.shape)

    return train_adl, X_teste, val_adl, y_teste


def run_ae(model, X, y, mse_thr):
    inv_trans = model.predict(X)

    mse = np.mean(np.power(X - inv_trans, 2), axis=1)
    pred = None
    pred = classify(mse, mse_thr)

    pred = np_utils.to_categorical(pred, n_classes)

    return pred, np.mean(mse)


def run(mse_thr, model='SVD'):
    test_adl = None
    test_adl_lbl = None
    fall = None
    fall_lbl = None
    X_teste = None

    if model == 'AE' or model == 'VAE':
        train_adl, X_teste, val_adl, y_teste = prep_ae(db)
    else:
        train_adl, train_adl_lbl, test_adl, test_adl_lbl, fall, fall_lbl = prep2()
        scaler = Normalizer()
        scaler.fit(train_adl)
        train_adl = scaler.transform(train_adl)
        test_adl = scaler.transform(test_adl)
        fall = scaler.transform(fall)

    svd = None

    if model == 'SVD':
        svd = TruncatedSVD(n_components=n_components, n_iter=100, random_state=42)
    elif model == 'PCA':
        svd = PCA(n_components=n_components)
    elif model == 'FOREST':
        svd = IsolationForest(contamination=mse_thr, behaviour='new')
    elif model == 'SVM':
        nu = fall.shape[0] / train_adl.shape[0]
        svd = OneClassSVM(nu=nu, kernel='rbf', gamma=n_components)
    elif model == 'AE':
        input_dim = X_teste.shape[1]
        hidden_dim = int(encoding_dim / 2)
        learning_rate = 1e-3
        if db == 'PRECIS_HAR':
            svd = get_model_precis_har(encoding_dim, hidden_dim, learning_rate)
        elif db == 'UpFall':
            svd = get_model_up_fall(encoding_dim, hidden_dim, learning_rate)
        else:
            svd = get_model_ur_fall()

    if model == 'AE' or model == 'VAE':

        if db == 'PRECIS_HAR':
            nb_epoch = 128
            batch_size = 64

        elif db == 'UpFall':
            nb_epoch = 128
            batch_size = 64
        else:
            nb_epoch = 200
            batch_size = 512

        svd.fit(train_adl, train_adl, verbose=1,
                epochs=nb_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(train_adl, train_adl))
    else:
        svd.fit(train_adl)

    pred_test_adl = None
    pred_fall = None
    pred = None

    if model in ['PCA', 'SVD']:
        trans = svd.transform(train_adl)
        inv_trans = svd.inverse_transform(trans)
        mse = np.mean(np.power(train_adl - inv_trans, 2), axis=1)
        if use_dynamic_threshold:
            mseTheshold = np.max(mse)  # + np.std(mse)
        else:
            mseTheshold = mse_thr

        pred_test_adl = run_svd(svd, test_adl, test_adl_lbl, 'TEST_ADL', mseTheshold)
        pred_fall = run_svd(svd, fall, fall_lbl, 'TEST_FALL', mseTheshold)
    elif model in ['FOREST', 'SVM']:
        yhat = svd.predict(test_adl)
        yhatFall = svd.predict(fall)

        yhat[yhat == 1] = 0
        yhat[yhat == -1] = 1

        yhatFall[yhatFall == 1] = 0
        yhatFall[yhatFall == -1] = 1

        pred_test_adl = np_utils.to_categorical(yhat, n_classes)
        pred_fall = np_utils.to_categorical(yhatFall, n_classes)
    else:
        trans = svd.predict(val_adl)
        mse_val_adl = np.mean(np.power(val_adl - trans, 2), axis=1)

        if db == 'PRECIS_HAR':
            mse_thr = 1.57
        elif db == 'UpFall':
            mse_thr = 1.128
        else:
            mse_thr = 1.12

        pred_teste, mse_teste = run_ae(svd, X_teste, y_teste, np.mean(mse_val_adl) * mse_thr)
        pred = np.argmax(pred_teste, axis=-1)
        true_lbl = y_teste

    if pred is None:
        pred = np.concatenate((pred_test_adl, pred_fall), axis=0)
        true_lbl = np.concatenate((test_adl_lbl, fall_lbl), axis=0)

    metrics = get_metrics(true_lbl, pred, cal_roc=True)
    exe.append(metrics)

    return metrics


dataset = './results-single/' + db + '-' + str(length_dataset)

exe = []

for x in range(number_of_experiments):
    metrics = run(mse_thr, model=model)
file = open(dataset + '-' + model + '-output_ALL.txt', 'w+')
file.write(str(exe))
