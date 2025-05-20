import pandas as pd

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
    sns.set_style("whitegrid") # 设置seaborn绘图风格
except ImportError:
    PLOT_AVAILABLE = False
    print("警告: 未找到 matplotlib 或 seaborn 库。将跳过可视化部分。")
    print("请使用 'pip install matplotlib seaborn' 命令安装它们以启用可视化。")

# 读取CSV文件
try:
    df = pd.read_csv('train_kill_nan.csv')
    print(f"数据集 train_kill_nan.csv 形状: {df.shape}")
    print("\n" + "="*50 + "\n")

    # 打印数据的前5行
    print("数据前5行:")
    print(df.head())
    print("\n" + "="*50 + "\n")

    # 打印数据的基本统计信息
    print("基本统计信息:")
    print(df.describe(include='all'))
    print("\n" + "="*50 + "\n")

    # 打印数据的列信息
    print("列信息 (数据类型, 非空值数量):")
    df.info()
    print("\n" + "="*50 + "\n")

    # ---- EDA 可视化 (如果库可用) ----
    if PLOT_AVAILABLE and not df.empty:
        print("\n" + "="*30 + " 清理后数据探索性分析可视化 (train_kill_nan.csv) " + "="*30)

        # 1. 目标变量 Survived 分布
        if 'Survived' in df.columns:
            plt.figure(figsize=(6, 4))
            sns.countplot(x='Survived', data=df)
            plt.title('目标变量 Survived 分布 (0=未生存, 1=生存) - train_kill_nan.csv')
            plt.xlabel('是否生存')
            plt.ylabel('数量')
            plt.show()
        else:
            print("警告: 'Survived' 列未找到，无法绘制其分布图。")

        vis_numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
        vis_categorical_features = ['Pclass', 'Sex', 'Embarked']

        # 2. 数值特征分布
        print("\n数值特征分布 (train_kill_nan.csv):")
        for col in vis_numerical_features:
            if col in df.columns:
                plt.figure(figsize=(8, 5))
                sns.histplot(df[col], kde=True) # train_kill_nan.csv should not have NaNs here
                plt.title(f'{col} 分布 (train_kill_nan.csv)')
                plt.xlabel(col)
                plt.ylabel('频率')
                plt.show()
            else:
                print(f"警告: 列 '{col}' 在数据集中未找到，跳过其分布图。")

        # 3. 分类特征分布
        print("\n分类特征分布 (train_kill_nan.csv):")
        for col in vis_categorical_features:
            if col in df.columns:
                plt.figure(figsize=(8, 5))
                sns.countplot(x=col, data=df, palette='viridis') # train_kill_nan.csv should not have NaNs here
                plt.title(f'{col} 分布 (train_kill_nan.csv)')
                plt.xlabel(col)
                plt.ylabel('数量')
                plt.show()
            else:
                print(f"警告: 列 '{col}' 在数据集中未找到，跳过其分布图。")
        
        # 4. 数值特征相关性热图
        actual_numerical_features_for_corr = [col for col in vis_numerical_features if col in df.columns]
        if actual_numerical_features_for_corr:
            plt.figure(figsize=(8, 6))
            correlation_matrix = df[actual_numerical_features_for_corr].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('数值特征相关性热图 (train_kill_nan.csv)')
            plt.show()
        else:
            print("警告: 没有可用于绘制相关性热图的数值特征。")

        # 5. 特征与 Survived 的关系
        print("\n特征与 Survived 的关系 (train_kill_nan.csv):")
        if 'Survived' not in df.columns:
            print("警告: 'Survived' 列未找到，无法绘制与生存状态相关的图表。")
        else:
            # Sex vs Survived
            if 'Sex' in df.columns:
                plt.figure(figsize=(6, 4))
                sns.countplot(x='Sex', hue='Survived', data=df, palette='muted')
                plt.title('性别 vs 生存状态 (train_kill_nan.csv)')
                plt.xlabel('性别')
                plt.ylabel('数量')
                plt.show()

            # Pclass vs Survived
            if 'Pclass' in df.columns:
                plt.figure(figsize=(6, 4))
                sns.countplot(x='Pclass', hue='Survived', data=df, palette='pastel')
                plt.title('船舱等级 vs 生存状态 (train_kill_nan.csv)')
                plt.xlabel('船舱等级')
                plt.ylabel('数量')
                plt.show()

            # Embarked vs Survived
            if 'Embarked' in df.columns:
                plt.figure(figsize=(6, 4))
                sns.countplot(x='Embarked', hue='Survived', data=df, palette='dark')
                plt.title('登船港口 vs 生存状态 (train_kill_nan.csv)')
                plt.xlabel('登船港口')
                plt.ylabel('数量')
                plt.show()
            
            # Age vs Survived
            if 'Age' in df.columns:
                plt.figure(figsize=(10, 6))
                sns.kdeplot(data=df, x='Age', hue='Survived', fill=True, common_norm=False, palette='rocket')
                plt.title('年龄分布 vs 生存状态 (train_kill_nan.csv)')
                plt.xlabel('年龄')
                plt.ylabel('密度')
                plt.show()

            # Fare vs Survived
            if 'Fare' in df.columns:
                plt.figure(figsize=(10, 6))
                sns.kdeplot(data=df, x='Fare', hue='Survived', fill=True, common_norm=False, palette='crest', log_scale=True)
                plt.title('票价 (对数刻度) 分布 vs 生存状态 (train_kill_nan.csv)')
                plt.xlabel('票价 (对数刻度)')
                plt.ylabel('密度')
                plt.show()
        
        print("\n" + "="*30 + " EDA 可视化结束 " + "="*30)
    # ---- EDA 可视化结束 ----

except FileNotFoundError:
    print("错误: 'train_kill_nan.csv' 文件未找到。请确保文件在脚本所在的目录下。")
except Exception as e:
    print(f"处理文件时发生错误: {e}") 