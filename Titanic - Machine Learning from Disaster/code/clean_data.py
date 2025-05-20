import pandas as pd

try:
    # 读取原始CSV文件
    df = pd.read_csv('train.csv')
    print(f"原始数据集形状: {df.shape}")

    # 删除所有包含缺失值的行
    df_cleaned = df.dropna()
    print(f"删除缺失值后的数据集形状: {df_cleaned.shape}")

    # 保存到新的CSV文件
    df_cleaned.to_csv('train_kill_nan.csv', index=False)
    print("已将处理后的数据保存到 'train_kill_nan.csv'")

except FileNotFoundError:
    print("错误: 'train.csv' 文件未找到。请确保文件在脚本所在的目录下。")
except Exception as e:
    print(f"处理文件时发生错误: {e}") 
