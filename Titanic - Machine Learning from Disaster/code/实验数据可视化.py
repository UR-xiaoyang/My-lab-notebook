import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
# 假设你已经加载了数据，并进行了必要的预处理
# 例如:
# train_df_original = pd.read_csv('train.csv')
# test_df_original = pd.read_csv('test.csv')

# 模拟一个经过预处理的数据 (你需要用你实际的数据替换)
# 这里的模拟数据只是为了让代码能跑通，不代表真实分布
data_sim = {
    'Survived': np.random.randint(0, 2, 100),
    'Pclass': np.random.randint(1, 4, 100),
    'Sex': np.random.choice(['male', 'female'], 100),
    'Age': np.random.normal(30, 15, 100),
    'Fare': np.random.lognormal(3, 1, 100),
    'Embarked': np.random.choice(['S', 'C', 'Q', np.nan], 100),
    'Cabin': [f'{c}{np.random.randint(1,100)}' if np.random.rand() > 0.7 else np.nan for c in np.random.choice(['A','B','C','D','E','F','G'], 100)],
    'Title': np.random.choice(['Mr', 'Miss', 'Mrs', 'Master'], 100),
    'FamilySize': np.random.randint(1, 8, 100)
}
train_df_original_sim = pd.DataFrame(data_sim)
train_df_original_sim['Age'] = train_df_original_sim['Age'].clip(0, 80)
train_df_original_sim['Fare'] = train_df_original_sim['Fare'].clip(0, 500)

# --- 图1：原始数据缺失值数量条形图 ---
plt.figure(figsize=(10, 6))
missing_values = train_df_original_sim.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
sns.barplot(x=missing_values.index, y=missing_values.values, palette="viridis")
plt.title('图1：原始数据缺失值数量 (模拟数据)', fontsize=15)
plt.xlabel('特征', fontsize=12)
plt.ylabel('缺失值数量', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
# plt.savefig('figure1_missing_values.png') # 取消注释以保存
# plt.show() # 在脚本中运行时使用

# --- 图2：票价 (Fare) 分布直方图/KDE图 ---
plt.figure(figsize=(10, 6))
sns.histplot(train_df_original_sim['Fare'].dropna(), kde=True, color="skyblue", bins=50)
plt.title('图2：票价 (Fare) 分布 (模拟数据)', fontsize=15)
plt.xlabel('票价', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.tight_layout()
# plt.savefig('figure2_fare_distribution.png')
# plt.show()

# --- 图3：性别 (Sex) 与生存率关系条形图 ---
plt.figure(figsize=(8, 5))
sns.barplot(x='Sex', y='Survived', data=train_df_original_sim, palette="pastel", ci=None)
plt.title('图3：性别与生存率关系 (模拟数据)', fontsize=15)
plt.xlabel('性别', fontsize=12)
plt.ylabel('平均生存率', fontsize=12)
plt.tight_layout()
# plt.savefig('figure3_sex_survival.png')
# plt.show()

# --- 图4：船舱等级 (Pclass) 与生存率关系条形图 ---
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Survived', data=train_df_original_sim, palette="muted", ci=None)
plt.title('图4：船舱等级与生存率关系 (模拟数据)', fontsize=15)
plt.xlabel('船舱等级', fontsize=12)
plt.ylabel('平均生存率', fontsize=12)
plt.tight_layout()
# plt.savefig('figure4_pclass_survival.png')
# plt.show()

# --- 图5：部分重要衍生特征（如Title, FamilySize）与生存率关系图 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(x='Title', y='Survived', data=train_df_original_sim, ax=axes[0], palette="Set2", ci=None)
axes[0].set_title('头衔 (Title) 与生存率 (模拟数据)', fontsize=13)
axes[0].tick_params(axis='x', rotation=45)
axes[0].set_xlabel('头衔', fontsize=11)
axes[0].set_ylabel('平均生存率', fontsize=11)

sns.barplot(x='FamilySize', y='Survived', data=train_df_original_sim, ax=axes[1], palette="Set3", ci=None)
axes[1].set_title('家庭规模 (FamilySize) 与生存率 (模拟数据)', fontsize=13)
axes[1].set_xlabel('家庭规模', fontsize=11)
axes[1].set_ylabel('平均生存率', fontsize=11)

fig.suptitle('图5：衍生特征与生存率关系', fontsize=16, y=1.03)
plt.tight_layout()
# plt.savefig('figure5_derived_features_survival.png')
# plt.show()


# --- 图6：主要模型Kaggle准确率对比条形图 ---
model_scores = {
    'LR (4.b)': 0.77751,
    'RF (4.c)': 0.75837,
    'LR (4.d)': 0.78229,
    'RF (4.e)': 0.77751,
    'LR (4.f)': 0.78468,
    'XGB (4.g)': 0.77990,
    'SVM (4.k)': 0.77272,
    'NN (4.l)': 0.78468,
    'NN (4.m)': 0.78708,
    'Ensemble (4.n)': 0.78708
}
models_df = pd.DataFrame(list(model_scores.items()), columns=['Model', 'Kaggle Accuracy']).sort_values(by='Kaggle Accuracy', ascending=False)

plt.figure(figsize=(12, 7))
sns.barplot(x='Kaggle Accuracy', y='Model', data=models_df, palette='viridis')
plt.title('图6：主要模型Kaggle准确率对比', fontsize=15)
plt.xlabel('Kaggle准确率', fontsize=12)
plt.ylabel('模型 (实验章节)', fontsize=12)
plt.xlim(0.70, 0.80) # Adjust x-axis limits if necessary
for index, value in enumerate(models_df['Kaggle Accuracy']):
    plt.text(value + 0.001, index, f'{value:.5f}')
plt.tight_layout()
# plt.savefig('figure6_model_comparison_kaggle.png')
# plt.show()

# --- 图7：最佳单模型（如4.m NN）与集成模型（4.n）的ROC曲线对比图（基于本地验证集） ---
# 你需要有本地验证集的真实标签 (y_val) 和模型预测概率
# 假设:
# y_val_nn_4m, y_pred_proba_nn_4m 来自4.m的神经网络
# y_val_ensemble_4n, y_pred_proba_ensemble_4n 来自4.n的集成模型

# 模拟数据，你需要替换为真实数据
y_val_sim = np.random.randint(0, 2, 50)
y_pred_proba_nn_sim = np.random.rand(50) * 0.4 + 0.3 # Simulate some reasonable probabilities
y_pred_proba_ensemble_sim = y_pred_proba_nn_sim * 0.9 + np.random.rand(50) * 0.1 # Ensemble often slightly better

# fpr_nn, tpr_nn, _ = roc_curve(y_val_nn_4m, y_pred_proba_nn_4m)
# roc_auc_nn = auc(fpr_nn, tpr_nn)
# fpr_ensemble, tpr_ensemble, _ = roc_curve(y_val_ensemble_4n, y_pred_proba_ensemble_4n)
# roc_auc_ensemble = auc(fpr_ensemble, tpr_ensemble)

fpr_nn_sim, tpr_nn_sim, _ = roc_curve(y_val_sim, y_pred_proba_nn_sim)
roc_auc_nn_sim = auc(fpr_nn_sim, tpr_nn_sim)
fpr_ensemble_sim, tpr_ensemble_sim, _ = roc_curve(y_val_sim, y_pred_proba_ensemble_sim)
roc_auc_ensemble_sim = auc(fpr_ensemble_sim, tpr_ensemble_sim)


plt.figure(figsize=(8, 6))
plt.plot(fpr_nn_sim, tpr_nn_sim, color='darkorange', lw=2, label=f'NN (4.m) ROC curve (AUC = {roc_auc_nn_sim:.4f}) (Simulated)')
plt.plot(fpr_ensemble_sim, tpr_ensemble_sim, color='green', lw=2, label=f'Ensemble (4.n) ROC curve (AUC = {roc_auc_ensemble_sim:.4f}) (Simulated)')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('图7：模型ROC曲线对比 (基于本地验证集 - 模拟数据)', fontsize=15)
plt.legend(loc="lower right")
plt.tight_layout()
# plt.savefig('figure7_roc_curves.png')
# plt.show()


# --- 图8：集成模型（4.n）中逻辑回归和XGBoost的特征重要性或系数图 ---
# 假设你已经训练好了逻辑回归模型 (lr_model_4n) 和 XGBoost模型 (xgb_model_4n)
# 并且有特征名称列表 (feature_names_4n)

# 模拟数据
# feature_names_sim = ['Age_scaled', 'Fare_log_scaled', 'FamilySize_scaled', 'Pclass_2', 'Pclass_3', 
#                  'Sex_male', 'Embarked_Q', 'Embarked_S', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs']
# lr_coefs_sim = np.random.randn(len(feature_names_sim)) * 0.5
# xgb_importances_sim = np.random.rand(len(feature_names_sim))
# xgb_importances_sim = xgb_importances_sim / xgb_importances_sim.sum() # Normalize


# # 逻辑回归系数 (对于独热编码后的特征)
# try:
#     if hasattr(lr_model_4n, 'coef_'):
#         coefficients = lr_model_4n.coef_[0]
#         lr_feature_importance = pd.DataFrame({'Feature': feature_names_4n, 'Coefficient': coefficients})
#         lr_feature_importance = lr_feature_importance.sort_values(by='Coefficient', key=abs, ascending=False).head(15) # Top 15 by absolute value

#         plt.figure(figsize=(10, 7))
#         sns.barplot(x='Coefficient', y='Feature', data=lr_feature_importance, palette='coolwarm')
#         plt.title('图8a：逻辑回归模型 (4.n) 特征系数 (模拟数据)', fontsize=15)
#         plt.xlabel('系数大小', fontsize=12)
#         plt.ylabel('特征', fontsize=12)
#         plt.tight_layout()
#         # plt.savefig('figure8a_lr_coefficients.png')
#         # plt.show()
# except NameError:
#     print("图8a：lr_model_4n 或 feature_names_4n 未定义，跳过逻辑回归系数图。请确保模型已训练并提供特征名。")


# # XGBoost 特征重要性
# try:
#     if hasattr(xgb_model_4n, 'feature_importances_'):
#         importances = xgb_model_4n.feature_importances_
#         xgb_feature_importance = pd.DataFrame({'Feature': feature_names_4n, 'Importance': importances})
#         xgb_feature_importance = xgb_feature_importance.sort_values(by='Importance', ascending=False).head(15)

#         plt.figure(figsize=(10, 7))
#         sns.barplot(x='Importance', y='Feature', data=xgb_feature_importance, palette='crest')
#         plt.title('图8b：XGBoost模型 (4.n) 特征重要性 (模拟数据)', fontsize=15)
#         plt.xlabel('重要性分数', fontsize=12)
#         plt.ylabel('特征', fontsize=12)
#         plt.tight_layout()
#         # plt.savefig('figure8b_xgb_importance.png')
#         # plt.show()
# except NameError:
#     print("图8b：xgb_model_4n 或 feature_names_4n 未定义，跳过XGBoost重要性图。请确保模型已训练并提供特征名。")
print("图8代码框架已提供，你需要用你实际训练的模型和特征名来生成。由于这里没有实际模型对象，暂时跳过绘图。")


# --- 图9：4.m神经网络训练过程中的学习曲线 ---
# 假设你有神经网络训练历史记录 `history_nn_4m`
# history_nn_4m = model.fit(...) # Keras返回的History对象
# 模拟数据
history_nn_4m_sim = {
    'accuracy': np.linspace(0.6, 0.85, 50) + np.random.rand(50)*0.05,
    'val_accuracy': np.linspace(0.62, 0.78, 50) + np.random.rand(50)*0.05,
    'loss': np.linspace(0.7, 0.4, 50) - np.random.rand(50)*0.05,
    'val_loss': np.linspace(0.68, 0.45, 50) - np.random.rand(50)*0.05
}
# history_nn_4m_sim['val_accuracy'][30:] -= np.linspace(0, 0.1, 20) # Simulate some overfitting
# history_nn_4m_sim['val_loss'][30:] += np.linspace(0, 0.1, 20)


# try:
#     # acc = history_nn_4m.history['accuracy']
#     # val_acc = history_nn_4m.history['val_accuracy']
#     # loss = history_nn_4m.history['loss']
#     # val_loss = history_nn_4m.history['val_loss']
#     # epochs_range = range(len(acc))

#     acc = history_nn_4m_sim['accuracy']
#     val_acc = history_nn_4m_sim['val_accuracy']
#     loss = history_nn_4m_sim['loss']
#     val_loss = history_nn_4m_sim['val_loss']
#     epochs_range = range(len(acc))


#     plt.figure(figsize=(14, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs_range, acc, label='训练集准确率 (Simulated)')
#     plt.plot(epochs_range, val_acc, label='验证集准确率 (Simulated)')
#     plt.legend(loc='lower right')
#     plt.title('图9a：神经网络 (4.m) 训练和验证准确率 (模拟数据)', fontsize=13)
#     plt.xlabel('轮次 (Epochs)', fontsize=11)
#     plt.ylabel('准确率', fontsize=11)

#     plt.subplot(1, 2, 2)
#     plt.plot(epochs_range, loss, label='训练集损失 (Simulated)')
#     plt.plot(epochs_range, val_loss, label='验证集损失 (Simulated)')
#     plt.legend(loc='upper right')
#     plt.title('图9b：神经网络 (4.m) 训练和验证损失 (模拟数据)', fontsize=13)
#     plt.xlabel('轮次 (Epochs)', fontsize=11)
#     plt.ylabel('损失', fontsize=11)
    
#     plt.suptitle('图9：神经网络学习曲线', fontsize=16, y=1.02)
#     plt.tight_layout()
#     # plt.savefig('figure9_nn_learning_curves.png')
#     # plt.show()
# except NameError:
#     print("图9：history_nn_4m 未定义，跳过学习曲线图。请确保模型已训练并提供Keras History对象。")
print("图9代码框架已提供，你需要用你实际神经网络的训练历史来生成。由于这里没有实际History对象，暂时跳过绘图。")

# 显示所有已生成的模拟图 (如果在Jupyter等环境中，可能不需要这行)
plt.show()