#导入必要的库
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV



print ("**********************************TUNE5*******************************")
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
    train = pd.read_csv('/home/zhenxing/DATA/CYP DATASET/Pubchem410/CYP1A2_PaDEL12_pcfp_train.csv')
    target = 'label'
    IDcol = 'Name'
    predictors = [x for x in train.columns if x not in [target, IDcol]]

    param_test5 = {'subsample':list([i/20 for i in range(12,19,1)])}

    gsearch5 = GridSearchCV( estimator = GradientBoostingClassifier( learning_rate=0.1, n_estimators=240,
                         max_depth=11, min_samples_split=100,min_samples_leaf=20,  max_features=46, random_state=10),
                         param_grid = param_test5,
                         scoring='roc_auc',
                         iid=False,verbose=1,
                         cv=5)
    gsearch5.fit(train[predictors],train[target])
    print_best_score(gsearch5,  param_test5)