import pandas as pd 
import numpy as np
import lightgbm as lgb 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

# 读取数据
def preProcess():
    path = 'data/'
    df_train = pd.read_csv(path + 'train.csv')
    df_test = pd.read_csv(path + 'test.csv')
    df_train.drop(['Id'], axis = 1, inplace = True)
    df_test.drop(['Id'], axis = 1, inplace = True)
    df_test['Label'] = -1
    data = pd.concat([df_train, df_test])
    data = data.fillna(-1)
    data.to_csv('data/data.csv', index = False)
    return data



def gbdt_lr_predict(data, category_feature, continuous_feature): 
    # 类别特征one-hot编码
    for col in category_feature:
        onehot_feats = pd.get_dummies(data[col], prefix = col)
        data.drop([col], axis = 1, inplace = True)
        data = pd.concat([data, onehot_feats], axis = 1)

    train = data[data['Label'] != -1]
    target = train.pop('Label')
    test = data[data['Label'] == -1]
    test.drop(['Label'], axis = 1, inplace = True)

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(train, target, test_size = 0.2, random_state = 2020)
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_val.shape)
    # print(y_val.shape)
    n_estimators = 32
    num_leaves = 64
    # 开始训练gbdt，使用32课树，每课树64个叶节点
    model = lgb.LGBMRegressor(objective='binary',
                            subsample= 0.8,
                            min_child_weight= 0.5,
                            colsample_bytree= 0.7,
                            num_leaves=num_leaves,
                            learning_rate=0.05,
                            n_estimators=n_estimators,
                            random_state = 2020,
                            verbose=0)
    model.fit(x_train, y_train,
            eval_set = [(x_train, y_train), (x_test, y_test)],
            eval_names = ['train', 'val'],
            eval_metric = 'binary_logloss'
            ) # verbose=0 非法参数，版本问题？

    # 得到每一条训练数据落在了每棵树的哪个叶子结点上
    # pred_leaf = True 表示返回每棵树的叶节点序号
    gbdt_feats_train = model.predict(train, pred_leaf = True)
    
    # 打印结果的 shape：
    print(gbdt_feats_train.shape)
    # 打印前5个数据：
    print(gbdt_feats_train[:5])

    # 同样要获取测试集的叶节点索引
    gbdt_feats_test = model.predict(test, pred_leaf = True)

    # 将 32 课树的叶节点序号构造成 DataFrame，方便后续进行 one-hot
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(n_estimators)]
    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns = gbdt_feats_name) 
    df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns = gbdt_feats_name)
    train_len = df_train_gbdt_feats.shape[0]
    data = pd.concat([df_train_gbdt_feats, df_test_gbdt_feats])
    # print(data.head())

    # 对每棵树的叶节点序号进行 one-hot
    for col in gbdt_feats_name:
        onehot_feats = pd.get_dummies(data[col], prefix = col)# data的每一列，也就是每一棵树的每一个节点，都做成one-hot
        # print(onehot_feats.head())
        data.drop([col], axis = 1, inplace = True)# 删除原来的此列
        data = pd.concat([data, onehot_feats], axis = 1) # 拼接基于此列生成的one-hot列
    # print(data.head())
    train = data[: train_len] # 0-1599行,因为刚才给和一起了，现在再拆分开
    # print(len(train))
    test = data[train_len:]# 1599-end行
    # print(len(test))
    # 划分 LR 训练集、验证集
    x_train, x_test, y_train, y_test = train_test_split(train, target, test_size = 0.3, random_state = 2018)
    # print(x_train.shape)
    # print(y_train.shape)

    # 开始训练lr
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    tr_logloss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
    print('tr-logloss: ', tr_logloss)
    val_logloss = log_loss(y_test, lr.predict_proba(x_test)[:, 1])
    print('val-logloss: ', val_logloss)
    # 对测试集预测，选取第二个类别的概率，默认是二分类，两个类别的概率，shape是（n_samples,2）
    y_pred_proba = lr.predict_proba(x_test)[:, 1]
    # print(x_test.shape)
    # print(y_pred)
    # 设置阈值（通常阈值为 0.5，但也可以调整）
    threshold = 0.5
    # 根据阈值判断最终预测类别
    y_pred = (y_pred_proba > threshold).astype(int)  # 如果概率大于阈值，预测为 1，否则为 0
    print(y_pred.shape)
    # 将结果调整为 (n_samples, 1) 的形状
    y_pred = y_pred.reshape(-1, 1)
    # print(y_test)
    print('accuracy:', accuracy_score(y_test,y_pred)*100)

    
if __name__ == '__main__':
    data = preProcess()

    # print(data.head())

    continuous_feature = ['I'] * 13
    continuous_feature = [col + str(i + 1) for i, col in enumerate(continuous_feature)]
    category_feature = ['C'] * 26
    category_feature = [col + str(i + 1) for i, col in enumerate(category_feature)]
    gbdt_lr_predict(data, category_feature, continuous_feature)