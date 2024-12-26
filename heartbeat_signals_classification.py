from matplotlib import rcParams
import warnings
from sklearn.model_selection import train_test_split
rcParams['font.family'] = 'simhei'
rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, auc, recall_score, f1_score, precision_score
import lightgbm as lgb
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import joblib
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
import streamlit as st

# %%
# 读取数据
# df = pd.read_csv('train.csv')
# df1 = pd.read_csv('testA.csv')
# # %%
# print(df.head(5), df1.head(5))
# # %%
# data_list = []
# for item in df.values:
#     data_list.append([item[0]] + [float(i) for i in item[1].split(',')] + [item[2]])
# data = pd.DataFrame(np.array(data_list))
# data.columns = ['id'] + ['heartbeat' + str(i) for i in range(len(data_list[0]) - 2)] + ['label']
#joblib.dump(data,'data.csv')
data = joblib.load('data.csv')

# %%
def reduce_mem_usage(df):
    # 处理前 数据集总内存计算
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    # 遍历特征列
    for col in df.columns:
        # 当前特征类型
        col_type = df[col].dtype
        # 处理 numeric 型数据
        if col_type != object:
            c_min = df[col].min()  # 最小值
            c_max = df[col].max()  # 最大值
            # int 型数据 精度转换
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                    # float 型数据 精度转换
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        # 处理 object 型数据
        else:
            df[col] = df[col].astype('category')  # object 转 category

    # 处理后 数据集总内存计算
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# %%
data = reduce_mem_usage(data)
# %%
print(data.head(5), data.describe(), data.info())
# %%
print(data.isnull().sum())
#%%
import seaborn as sns
sns.boxplot((data.drop(['id','label'],axis= 1)))
# %%
# 查看重复值
duplicates = data[data.duplicated()]
print(duplicates)
# %%
fig = plt.figure()
plt.hist(data['label'], orientation='vertical', histtype='bar')
plt.show()
# %%
x = data.drop(['id', 'label'], axis=1)
y = data['label']
print(x.shape, y.shape)
# %%
target_count = np.bincount(y)
somte = SMOTE(sampling_strategy={1: target_count[3]}, random_state=42)
x_resample, y_resample = somte.fit_resample(x, y)
# %%
print(np.bincount(y_resample))
# %%
x_train, x_test, y_train, y_test = train_test_split(x_resample, y_resample, train_size=0.8, random_state=10)
# %%
# model_list = [MultinomialNB(),
#               DecisionTreeClassifier(criterion='entropy', max_depth=5, class_weight='balanced'),
#               SVC(probability=True, kernel='rbf', class_weight='balanced')]
# model_names = ['Naive_Bayes', 'Decision_Tree', 'SVC']
#
# for name, model in zip(model_names, model_list):
#     # model.fit(x_train, y_train)
#     model = joblib.load(f'{name}.pkl')
#     y_predict = model.predict(x_test)
#     # joblib.dump(model,'{}.pkl'.format(name))
#     print('{}分类模型在测试集上的评价结果为：'.format(name))
#     print(classification_report(y_test, y_predict))


# %%
def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(4, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='macro')
    return 'f1_score', score_vali, True


class_weights = compute_sample_weight('balanced', y_train)
train_matrix = lgb.Dataset(x_train, label=y_train, weight=class_weights)
test_matrix = lgb.Dataset(x_test, label=y_test)
# params = {
#     "learning_rate": 0.1,
#     "boosting": 'gbdt',
#     "lambda_l2": 0.1,
#     "max_depth": -1,
#     "num_leaves": 128,
#     "bagging_fraction": 0.8,
#     "feature_fraction": 0.8,
#     "metric": None,
#     "objective": "multiclass",
#     "num_class": 4,
#     "nthread": 10,
#     "verbose": -1,
# }

# """使用训练集数据进行模型训练"""
# model = lgb.train(params,
#                   train_set=train_matrix,
#                   valid_sets=test_matrix,
#                   num_boost_round=2000,
#                   feval=f1_score_vali
#                  )

# y_pred = model.predict(x_test,num_iteration=model.best_iteration)
# y_pred = np.argmax(y_pred, axis=1)
# print(classification_report(y_pred,y_test))
# %%
# """定义优化函数"""
# def rf_cv_lgb(num_leaves, max_depth, bagging_fraction, feature_fraction, bagging_freq, min_data_in_leaf,
#               min_child_weight, min_split_gain, reg_lambda, reg_alpha):
#     # 建立模型
#     model_lgb = lgb.LGBMClassifier(boosting_type='gbdt', objective='multiclass', num_class=4,
#                                    learning_rate=0.1, n_estimators=5000,
#                                    num_leaves=int(num_leaves), max_depth=int(max_depth),
#                                    bagging_fraction=round(bagging_fraction, 2), feature_fraction=round(feature_fraction, 2),
#                                    bagging_freq=int(bagging_freq), min_data_in_leaf=int(min_data_in_leaf),
#                                    min_child_weight=min_child_weight, min_split_gain=min_split_gain,
#                                    reg_lambda=reg_lambda, reg_alpha=reg_alpha,
#                                    n_jobs= 8
#                                   )
#     f1 = make_scorer(f1_score, average='micro')
#     val = cross_val_score(model_lgb, x_train, y_train, cv=5, scoring=f1).mean()

#     return val
# #"""定义优化参数"""
# bayes_lgb = BayesianOptimization(
#     rf_cv_lgb,
#     {
#         'num_leaves':(10, 200),
#         'max_depth':(3, 20),
#         'bagging_fraction':(0.5, 1.0),
#         'feature_fraction':(0.5, 1.0),
#         'bagging_freq':(0, 100),
#         'min_data_in_leaf':(10,100),
#         'min_child_weight':(0, 10),
#         'min_split_gain':(0.0, 1.0),
#         'reg_alpha':(0.0, 10),
#         'reg_lambda':(0.0, 10),
#     }
# )

# #"""开始优化"""
# bayes_lgb.maximize(n_iter=10)

# params = bayes_lgb.max
# print(params)
# joblib.dump(params,'params.pkl')
# %%
base_params_lgb = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 4,
    'learning_rate': 0.01,
    'num_leaves': 84,
    'max_depth': 19,
    'min_data_in_leaf': 78,
    'min_child_weight': 7.7,
    'bagging_fraction': 0.57,
    'feature_fraction': 0.63,
    'bagging_freq': 61,
    'reg_lambda': 0.4,
    'reg_alpha': 1.13,
    'min_split_gain': 0.05,
    'nthread': 10,
    'verbose': -1,
}

# cv_result_lgb = lgb.cv(
#     train_set=train_matrix,
#     num_boost_round=20000,
#     nfold=5,
#     stratified=True,
#     shuffle=True,
#     params=base_params_lgb,
#     feval=f1_score_vali,
#     seed=0
# )
# print('迭代次数{}'.format(len(cv_result_lgb['f1_score-mean'])))
# print('最终模型的f1为{}'.format(max(cv_result_lgb['f1_score-mean'])))
# %%
model = joblib.load('LightGBM.pkl')
# model = lgb.train(base_params_lgb, train_set=train_matrix, num_boost_round=4833,
#                   feval=f1_score_vali)
y_pred_prob = model.predict(x_test, num_iteration=model.best_iteration)
y_pred = np.argmax(y_pred_prob, axis=1)
# joblib.dump(model,'LightGBM.pkl')
# %%
classification_report = classification_report(y_pred, y_test,output_dict=True)
from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

# %%
st.sidebar.image("logo.png")
st.sidebar.title("基于LightGBM的心跳信号分类预测")
st.sidebar.write('Made By FanRou yi')
st.sidebar.markdown('------')
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fredoka+One&display=swap');

        .artistic-title {
            font-family: 'Fredoka One';
            font-size: 70px;

            color: #FE4676;
            text-align: center;
            text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.4);
            letter-spacing: 2px;
            margin-top: 30px;
        }
    </style>
    <div class="artistic-title">
        Haertbeat Signals 
            Classification
    </div>
""", unsafe_allow_html=True)
with st.expander("📖有关这个streamlit app的详细介绍"):
    st.write("你好！欢迎来到Heartbeat Signals Classification。🤚\n\n"
             "这是一个基于LightGBM的心跳信号分类预测模型的streamlit app，app中的功能包括：进行样本预测，查看模型性能，查看模型训练源码。\n\n"
             "你可以输入一个样本的样本特征来进行预测，但请输入以逗号为间隔的205个特征，这样才可以顺利进行预测，我们会输出给你预测结果。\n\n"
             "你也可以查看本系统所使用的模型评估指标，我们会显示模型的分类报告，其中包含对每个标签预测的查准率，查全率，f1等等，还有每个标签的ROC图。\n\n"
             "此外，你还可以获取本项目的源码，里面包含有对模型的选择，训练，以及部署过程，还有模型训练所使用的数据来源。")
st.markdown("""
    <style>
        /* 定义侧边栏目录项的样式 */
        .sidebar-link {
            display: block;
            padding: 12px;
            margin: 8px 0;
            border: 1.7px solid #FE4676;  /* 设置边框颜色 */
            border-radius: 8px;         /* 设置圆角 */
            background-color: #f1f1f1;  /* 设置背景色 */
            text-decoration: none;      /* 去除链接下划线 */
            text-align: center;
            color:#4E4E4E ;             /* 设置文字颜色 */
            font-weight: bold;          /* 设置字体加粗 */
        }


        /* 给内容部分增加一点间距 */
        .content-section {
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<a href="#section-1" class="sidebar-link">💻样本预测</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#section-2" class="sidebar-link">🔎模型性能</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#section-3" class="sidebar-link">🚚代码和数据来源</a>', unsafe_allow_html=True)

# 主栏内容区域
st.markdown('<div id="section-1"></div>', unsafe_allow_html=True)
st.markdown("## 💻样本预测")
model = joblib.load('LightGBM.pkl')
user_input = st.text_input("请输入一组以逗号为间隔的样本特征", "")
if st.button("🎯 预测"):
    if user_input:
        try:
            input_features = np.array([float(x) for x in user_input.split(',')])
            input_x = pd.DataFrame(input_features.reshape(-1, 1))
            prediction_prob = model.predict(input_x[0], predict_disable_shape_check=True)
            prediction = np.argmax(prediction_prob, axis=1)
            st.write(prediction)
        except ValueError:
            st.error("输入无效，请确保每个特征值是数字并以逗号分隔。")
    else:
        st.error("请输入特征值进行预测。")

st.markdown('<div id="section-2"></div>', unsafe_allow_html=True)
st.markdown("## 🔎模型性能")
st.subheader("💌 Classification Report")
report_matrix = pd.DataFrame(classification_report).transpose()
st.dataframe(report_matrix)
st.subheader('📈 每个类别的ROC图')
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = ['blue', 'red', 'green', 'black']
with st.container():
    col1,col2 = st.columns(2)
    col3, col4 = st.columns(2)
    cols = [col1, col2, col3, col4]
    for i in range(4):
        with cols[i]:
            ax = plt.figure()
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'
                                                                  ''.format(i, roc_auc[i]))
            plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
            plt.legend(loc='lower right')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'label{i} ROC Curve')
            plt.show()
            st.pyplot(ax)

st.markdown('<div id="section-3"></div>', unsafe_allow_html=True)
st.markdown("## 🚚代码和数据来源")
with st.expander("📋程序源码"):
    st.markdown("[☁️本程序的代码，内容包含数据处理，模型选择，模型训练和前端部署](https://github.com/fanrouyi/heartbeat-signals-classification.git)")
with st.expander("📂数据来源"):
    st.markdown(
        "[☁️模型训练使用的数据来源为天池数据集中的心跳信号数据集，本程序的模型使用了其中十万条样本](https://tianchi.aliyun.com/dataset/167192)")

