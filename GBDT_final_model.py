import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import  metrics


Trainset = '/home/zhenxing/DATA/CYP DATASET/Pubchem884/CYP3A4_PaDEL12_pcfp_train.csv'
Testset = '/home/zhenxing/DATA/CYP DATASET/Pubchem884/CYP3A4_PaDEL12_pcfp_test.csv'
Learning_rate = 0.1
N_estimators = 240


print ("**********************************learning_rate=",Learning_rate,"**","n_estimators=",N_estimators,"*******************************")
#定义评估系数计算函数
def SPSEQMCC(label, prediction):
    TP, FN, TN, FP = 0, 0, 0, 0
    for i in range(len(label)):
        if label[i] == 0:
            if prediction[i] == 0:
                TN = TN + 1
            else:
                FP = FP + 1
        else:
            if prediction[i] == 0:
                FN = FN + 1
            else:
                TP = TP + 1
    SE = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    print("TP,FN,TN,FP:", TP, FN, TN, FP)
    print("ACC：", '%.3f'%ACC)
    print("SE：", '%.3f'%SE)
    print("SP：", '%.3f'%SP)
    L = (TN + FN) * (TN + FP) * (TP + FN) * (TP + FP)
    if L == 0:
        print("No MCC")
    else:
        MCC = (TP * TN - FP * FN) / (((TN + FN) * (TN + FP) * (TP + FN) * (TP + FP)) ** 0.5)
        print("MCC:", '%.3f'% MCC)


#定义构建模型的函数
def modelfit(alg, dtrain,dtest):
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Predict test set:
    dtest_predictions = alg.predict(dtest[predictors])
    dtest_predprob = alg.predict_proba(dtest[predictors])[:, 1]


    # Print model report:
    print ("\nModel Report (Train):")
    SPSEQMCC(dtrain[target].values, dtrain_predictions)
    print ("AUC Score (Train): %.3f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))

    print("\nModel Report (Test):")
    SPSEQMCC(dtest[target].values, dtest_predictions)
    print("AUC Score (Test): %.3f" % metrics.roc_auc_score(dtest[target], dtest_predprob))


#读入数据
train = pd.read_csv(Trainset)
test = pd.read_csv(Testset)
target = 'label'
IDcol = 'Name'
predictors = [x for x in train.columns if x not in [target, IDcol]]

#搭建模型
xgb1 = GradientBoostingClassifier(learning_rate=Learning_rate, n_estimators=N_estimators,
                                  max_depth=5, min_samples_split=1100,
                                  min_samples_leaf=40,  max_features=34, subsample=0.9,
                                  random_state=10)
modelfit(xgb1, train, test)