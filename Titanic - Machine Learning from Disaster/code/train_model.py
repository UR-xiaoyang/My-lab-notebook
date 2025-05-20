import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold # 导入 GridSearchCV 和 StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder # PolynomialFeatures (可选)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, make_scorer # 导入 make_scorer

# --- 1. 数据加载 ---
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    submission_df = pd.read_csv('gender_submission.csv')
except FileNotFoundError:
    print("错误：train.csv, test.csv 或 gender_submission.csv 文件未找到。请确保这些文件在脚本所在目录下。")
    exit()

combined_df = pd.concat([train_df.drop('Survived', axis=1), test_df], ignore_index=True)
y_train_full = train_df['Survived'] # 这是我们要预测的目标变量

# --- 2. 特征工程 ---
# (与你之前的代码相同，此处省略以保持简洁，实际使用时请保留这部分代码)
# 2.a. Title 特征
combined_df['Title'] = combined_df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
combined_df['Title'] = combined_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
combined_df['Title'] = combined_df['Title'].replace('Mlle', 'Miss')
combined_df['Title'] = combined_df['Title'].replace('Ms', 'Miss')
combined_df['Title'] = combined_df['Title'].replace('Mme', 'Mrs')
common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
combined_df.loc[~combined_df['Title'].isin(common_titles + ['Rare']), 'Title'] = 'Mr'

# 2.b. FamilySize 特征
combined_df['FamilySize'] = combined_df['SibSp'] + combined_df['Parch'] + 1

# 2.c. Deck 特征
combined_df['Deck'] = combined_df['Cabin'].str[0].fillna('U')

# 2.d. Age 和 Fare 的缺失值填充
combined_df['Age'] = combined_df['Age'].fillna(combined_df['Age'].median())
combined_df['Fare'] = combined_df['Fare'].fillna(combined_df['Fare'].median())

# 2.e. AgeBin 特征
age_bins = [0, 12, 18, 35, 60, np.inf]
age_labels = ['Child', 'Teenager', 'Adult', 'MiddleAged', 'Senior']
combined_df['AgeBin'] = pd.cut(combined_df['Age'], bins=age_bins, labels=age_labels, right=False)
condition_master_adult_or_senior = (combined_df['Title'] == 'Master') & (combined_df['AgeBin'].isin(['Adult', 'MiddleAged', 'Senior']))
combined_df.loc[condition_master_adult_or_senior, 'AgeBin'] = 'Teenager'

# 2.f. FareBin 特征
try:
    combined_df['FareBin'] = pd.qcut(combined_df['Fare'], 4, labels=['LowFare', 'MedFare', 'HighFare', 'VeryHighFare'], duplicates='drop')
except ValueError:
    print("Warning: qcut for FareBin failed, possibly due to insufficient distinct values. Using fixed bins as a fallback.")
    fare_bins_fixed = [0, combined_df['Fare'].quantile(0.25), combined_df['Fare'].quantile(0.5), combined_df['Fare'].quantile(0.75), np.inf]
    fare_labels_fixed = ['LowFare', 'MedFare', 'HighFare', 'VeryHighFare']
    for i in range(1, len(fare_bins_fixed)):
        if fare_bins_fixed[i] <= fare_bins_fixed[i-1]:
            fare_bins_fixed[i] = fare_bins_fixed[i-1] + 0.01
    if len(np.unique(fare_bins_fixed)) -1 < len(fare_labels_fixed):
         combined_df['FareBin'] = pd.cut(combined_df['Fare'], bins=len(fare_labels_fixed), labels=fare_labels_fixed, include_lowest=True, duplicates='drop')
    else:
        combined_df['FareBin'] = pd.cut(combined_df['Fare'], bins=fare_bins_fixed, labels=fare_labels_fixed, include_lowest=True, duplicates='drop')

# 2.g. 交互特征
combined_df['AgeBin_Pclass'] = combined_df['AgeBin'].astype(str) + "_" + combined_df['Pclass'].astype(str)
combined_df['Title_Pclass'] = combined_df['Title'].astype(str) + "_" + combined_df['Pclass'].astype(str)
combined_df['Sex_Pclass'] = combined_df['Sex'].astype(str) + "_" + combined_df['Pclass'].astype(str)

# 2.h. Embarked 缺失值填充
combined_df['Embarked'] = combined_df['Embarked'].fillna(combined_df['Embarked'].mode()[0])


# --- 3. 特征选择与类型定义 ---
numerical_features = ['FamilySize']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'AgeBin', 'FareBin',
                        'AgeBin_Pclass', 'Title_Pclass', 'Sex_Pclass']
features_to_use = numerical_features + categorical_features

# 使用整个训练集进行特征工程，然后在GridSearchCV中进行分割和交叉验证
X_for_gridsearch = combined_df.loc[:len(train_df)-1, features_to_use].copy() # 注意这里索引要对齐
y_for_gridsearch = y_train_full.copy()

# --- 4. 构建预处理和模型管道 ---
# 数值特征处理
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 分类特征处理
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

# 列转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ],
    remainder='passthrough' # 或者 'drop' 如果确定没有其他列
)

# 将预处理器和逻辑回归模型组合成一个完整的管道
# 这对于GridSearchCV非常重要，因为它会在交叉验证的每一折中独立地拟合预处理器
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)) # 增加max_iter以防不收敛
])

# --- 5. 超参数调优 (GridSearchCV) ---
print("开始超参数调优 (GridSearchCV)...")

# 定义要搜索的参数网格
# 注意：不同的solver支持不同的penalty
# 例如，'liblinear' 支持 'l1' 和 'l2'
# 'lbfgs', 'newton-cg', 'sag' 只支持 'l2' 或 None
# 'saga' 支持 'elasticnet', 'l1', 'l2', None
param_grid = [
    {
        'classifier__solver': ['liblinear'],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
    },
    {
        'classifier__solver': ['saga'], # saga 通常对大数据集和高维数据表现好
        'classifier__penalty': ['l1', 'l2', 'elasticnet'],
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__l1_ratio': [0.3, 0.5, 0.7] # 仅当 penalty='elasticnet' 时使用
    }
]
# 为了快速演示，可以简化param_grid:
# param_grid = {
#     'classifier__solver': ['liblinear', 'saga'],
#     'classifier__penalty': ['l1', 'l2'],
#     'classifier__C': [0.01, 0.1, 1, 10]
# }


# 使用分层K折交叉验证，确保每折中类别比例相似
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 定义评分标准，例如准确率或F1分数
# 如果需要概率输出进行log_loss评估，需要 make_scorer 和 predict_proba
# accuracy_scorer = make_scorer(accuracy_score)
# logloss_scorer = make_scorer(log_loss, needs_proba=True, greater_is_better=False) # log_loss越小越好

# GridSearchCV 会自动使用模型的 score 方法（对于 LogisticRegression 是准确率）
# 如果想用log_loss, 需要指定 scoring=logloss_scorer
grid_search = GridSearchCV(full_pipeline, param_grid, cv=cv_strategy,
                           scoring='accuracy', # 或者 scoring=logloss_scorer
                           verbose=1, n_jobs=-1) # n_jobs=-1 使用所有可用CPU核心

# 在整个训练数据上运行 GridSearchCV (它内部会进行交叉验证)
grid_search.fit(X_for_gridsearch, y_for_gridsearch)

print("\nGridSearchCV 完成。")
print("最佳参数组合: ", grid_search.best_params_)
print("交叉验证最佳准确率: {:.5f}".format(grid_search.best_score_))

# 获取最佳模型
best_lr_model = grid_search.best_estimator_

# --- 6. 使用最佳模型在验证集上评估 (可选步骤，因为GridSearchCV已经给出了交叉验证分数) ---
# 如果你仍然想保留一个独立的验证集来评估最终选定的模型
# X_train, X_val, y_train, y_val = train_test_split(
#     X_for_gridsearch, y_for_gridsearch, test_size=0.2, random_state=42, stratify=y_for_gridsearch
# )
# best_lr_model.fit(X_train, y_train) # 用最佳参数在更大的训练集上重新训练
# y_pred_val_probs = best_lr_model.predict_proba(X_val)[:, 1]
# y_pred_val_labels = best_lr_model.predict(X_val)
# loss_val = log_loss(y_val, y_pred_val_probs)
# accuracy_val = accuracy_score(y_val, y_pred_val_labels)
# print(f"\n--- 最佳逻辑回归模型在独立验证集上 ---")
# print(f"损失 (LogLoss): {loss_val:.4f}")
# print(f"准确率: {accuracy_val:.5f}")


# --- 7. 错误分析 (使用最佳模型在整个训练数据上训练后的预测，或交叉验证的预测) ---
# 为了简化，我们这里直接使用最佳模型在整个X_for_gridsearch上预测来进行错误分析
# 注意：这时的预测是在“见过”的数据上做的，仅用于分析特征，不代表泛化能力
# 更严谨的错误分析应该基于交叉验证中每一折的 hold-out set 预测结果
y_pred_full_train_labels = best_lr_model.predict(X_for_gridsearch) # best_lr_model 已经是 GridSearchCV 在整个数据上用最佳参数训练的结果
misclassified_indices = np.where(y_for_gridsearch.to_numpy() != y_pred_full_train_labels)[0]

if len(misclassified_indices) > 0:
    print("\n--- 详细错误分析：基于最佳模型在训练数据上错误分类的样本 ---")
    print(f"总共错误分类的样本数: {len(misclassified_indices)} / {len(X_for_gridsearch)}")

    # X_for_gridsearch 是未经过预处理的原始特征
    misclassified_original_features = X_for_gridsearch.iloc[misclassified_indices]
    true_labels_misclassified = y_for_gridsearch.iloc[misclassified_indices]
    predicted_labels_misclassified = pd.Series(y_pred_full_train_labels[misclassified_indices], index=true_labels_misclassified.index, name="PredictedSurvived")

    misclassified_df_display = pd.concat([misclassified_original_features, true_labels_misclassified.rename('TrueSurvived'), predicted_labels_misclassified], axis=1)

    print("部分错误分类的样本 (最多显示20条):")
    print(misclassified_df_display.head(20))

    misclassified_df_display.to_csv('misclassified_training_samples_best_lr.csv', index=True)
    print("所有在训练数据上错误分类的样本已保存到 'misclassified_training_samples_best_lr.csv'")
else:
    print("\n--- 详细错误分析 ---")
    print("训练数据上没有错误分类的样本 (可能过拟合，或模型完美)。")


# --- 8. 生成Kaggle提交文件 ---
print("\n开始为 test.csv 生成预测结果...")
# 从 combined_df 中提取对应于原始 test_df 的数据部分
# 这些数据已经经过了与训练数据相同的特征工程步骤
X_test_kaggle_raw = combined_df.iloc[len(train_df):][features_to_use].copy() # 应用相同的特征选择

# 使用最佳模型进行预测 (它内部包含了预处理器)
y_pred_test_kaggle = best_lr_model.predict(X_test_kaggle_raw)

submission = pd.DataFrame({'PassengerId': submission_df['PassengerId'], 'Survived': y_pred_test_kaggle})
submission['Survived'] = submission['Survived'].astype(int)
submission.to_csv('submission_best_lr.csv', index=False)
print("Kaggle提交文件 'submission_best_lr.csv' 已生成。")

# --- 补充：获取独热编码后的特征名称和模型系数 (可选) ---
try:
    # 获取预处理器和分类器步骤
    preprocessor_step = best_lr_model.named_steps['preprocessor']
    classifier_step = best_lr_model.named_steps['classifier']

    # 获取独热编码后的特征名
    onehot_encoder = preprocessor_step.named_transformers_['cat'].named_steps['onehot']
    if hasattr(onehot_encoder, 'get_feature_names_out'):
        cat_feature_names_out = onehot_encoder.get_feature_names_out(categorical_features)
    else: # 兼容旧版本
        cat_feature_names_out = []
        for i, cat_feat in enumerate(categorical_features):
            for cat_val in onehot_encoder.categories_[i]:
                if pd.isna(cat_val): cat_name = f"{cat_feat}_nan"
                else: cat_name = f"{cat_feat}_{cat_val}"
                cat_feature_names_out.append(cat_name)
    
    all_feature_names = numerical_features + list(cat_feature_names_out)
    
    # 获取模型系数
    coefficients = classifier_step.coef_[0] # 对于二分类，coef_ 是 (1, n_features)
    
    feature_importance = pd.DataFrame({'Feature': all_feature_names, 'Coefficient': coefficients})
    feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values(by='Abs_Coefficient', ascending=False)
    
    print("\n--- 特征重要性 (基于逻辑回归系数的绝对值) ---")
    print(feature_importance.head(15))
    feature_importance.to_csv('feature_importance_lr.csv', index=False)
    print("特征重要性已保存到 'feature_importance_lr.csv'")

except Exception as e:
    print(f"获取特征名称或系数时出错: {e}")

print("\n代码执行完毕。请检查输出的准确率和最佳参数。")