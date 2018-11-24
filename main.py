from sklearn.ensemble import RandomForestRegressor
from data_loader import VAT_DataSet
import numpy as np
from scipy.stats import pearsonr
import numpy
import time
import sys
from sklearn.model_selection import GridSearchCV
import argparse
import pickle
def ccc(y_true, y_pred):
    true_mean = numpy.mean(y_true)
    true_variance = numpy.var(y_true)
    pred_mean = numpy.mean(y_pred)
    pred_variance = numpy.var(y_pred)

    rho,_ = pearsonr(y_pred,y_true)

    std_predictions = numpy.std(y_pred)

    std_gt = numpy.std(y_true)


    ccc = 2 * rho * std_gt * std_predictions / (
        std_predictions ** 2 + std_gt ** 2 +
        (pred_mean - true_mean) ** 2)

    return ccc, rho


def cv_on_RF(X_train, y_train, X_test, y_test):
    ticks = time.time()
    estimator = RandomForestRegressor(random_state=220)

    # classifications
    param_grid = {
            #'n_estimators': [20, 50],}
            'max_features': ['auto', 'sqrt', 'log2', None],}
            #'criterion':['mse', 'mae'],}
            #'min_samples_split': [10, 30, 50]}
    score = 'neg_mean_absolute_error'
    print("# Tuning hyper-parameters for %s" % score)
    clf = GridSearchCV(estimator, param_grid, cv=5, scoring='%s' % score)
    clf.fit(X_train, y_train)
    print('Time Elapse: {}'.format(time.time()-ticks))
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Best scores found on development set:")
    print()
    print(clf.best_score_)
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print("CCC on val set:", ccc(y_true, y_pred)[0])
    ccc_value =  ccc(y_true, y_pred)[0]
    store_name = 'RF_ccc:{:.4f}_label:{}'.format(ccc_value, args.label_name)
    pickle.dump(clf, open(store_name+'.pkl', 'wb'))
    
def cv_on_xgb(X_train, y_train, X_test, y_test):
    import xgboost as xgb
    
    #first train with default parameters
    print("training with xgboost...")
    xgb_reg = xgb.XGBRegressor(random_state = 220)
    ticks = time.time()
    xgb_reg.fit(X_train, y_train)
    print('Time Elapse: {}'.format(time.time()-ticks))
    y_pred, y_true = y_test, xgb_reg.predict(X_test)
    print("CCC on val set:", ccc(y_true, y_pred)[0])
    ccc_value =  ccc(y_true, y_pred)[0]
    store_name = 'xgb_ccc:{:.4f}_label:{}'.format(ccc_value, args.label_name)
    setattr(xgb_reg, 'meta', meta)
    pickle.dump(xgb_reg, open(store_name+'.pkl', 'wb'))

def remove_no_feature_importance(pretrained_model, X_train, y_train, X_test, y_test):
    model = pickle.load(open(pretrained_model, 'rb'))
    mask= model.feature_importances_!=0.0
    
    X_train, y_train = X_train[:,mask], y_train
    X_test, y_test = X_test[:,mask], y_test
    
    return X_train, y_train, X_test, y_test

        
def main():
    X, y = [], []
    visual_root_path = '/newdisk/OMGEmotionChallenge/fur_experiment/transfer_learning/Extracted_features/vgg_fer_features_fps=15_fc7'
    audio_root_path = '/newdisk/OMGEmotionChallenge/fur_experiment/transfer_learning/Extracted_features/egemaps_VAD'
    text_root_path = '/newdisk/OMGEmotionChallenge/fur_experiment/transfer_learning/Extracted_features/MPQA' 
    
    train_dataset = VAT_DataSet(visual_root_path, audio_root_path, text_root_path, '../train_dict.pkl',  label_name=args.label_name)
    for i in range(len(train_dataset)):
        print("loading dataset : [{}/ {} ] \r".format(i, len(train_dataset)))
        sys.stdout.flush()
        X.append(train_dataset[i][0])
        y.append(train_dataset[i][1])
    X, y = np.asarray(X), np.asarray(y)
    # standarization
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_train, y_train = (X-mean)/std, y
    meta = {}
    meta['mean'] = mean
    meta['std'] = std
    meta['input_size'] = X_train.shape[1:]
    global meta
    X, y = [], []
    val_dataset = VAT_DataSet(visual_root_path, audio_root_path, text_root_path, '../val_dict.pkl',  label_name=args.label_name)
    for i in range(len(val_dataset)):
        print("loading dataset : [{}/ {} ] \r".format(i, len(val_dataset)))
        sys.stdout.flush()
        X.append(val_dataset[i][0])
        y.append(val_dataset[i][1])
    X, y = np.asarray(X), np.asarray(y)
    X_val, y_val = (X-mean)/std, y

    if args.classifier == 'rf':
        cv_on_RF(X_train, y_train,X_val, y_val )
    elif args.classifier == 'xgb':
        cv_on_xgb(X_train, y_train,X_val, y_val )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='decision tree Example')
    parser.add_argument('--label_name', type=str, default='arousal', 
                        help='label name')
    parser.add_argument('--classifier', type=str, default = 'rf',
                        help= 'random forest (rf) or xgboost (xgb)')
    args = parser.parse_args()
    global args
    main()
    