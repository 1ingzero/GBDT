#导入必要的库
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV

print ("**********************************TUNE1*******************************")

#定义输出寻优结果函数
def print_best_score(gsearch, param_test):
    # 输出best score
    print("grid_scores:",gsearch.grid_scores_)
    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    # 输出最佳的分类器到底使用了怎样的参数
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(param_test.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

#定义'max_depth'和'min_child_weight'的寻优函数
if __name__=='__main__':
    train = pd.read_csv('/home/zhenxing/DATA/CYP DATASET/Pubchem899/CYP2C19_PaDEL12_pcfp_train.csv')
    test = pd.read_csv('/home/zhenxing/DATA/CYP DATASET/Pubchem899/CYP2C19_PaDEL12_pcfp_test.csv')
    target = 'label'
    IDcol = 'Name'
    predictors = [x for x in train.columns if x not in [target, IDcol]]

    param_test1 = {'n_estimators':list(range(20,81,10))}

    gsearch1 = GridSearchCV( estimator = GradientBoostingClassifier( learning_rate=0.1, min_samples_split=300,
                         min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10),
                         param_grid = param_test1,
                         scoring='roc_auc',
                         iid=False,verbose=1,
                         cv=5)
    gsearch1.fit(train[predictors],train[target])
    print_best_score(gsearch1,param_test1)