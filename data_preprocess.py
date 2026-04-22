# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_titanic(train_path='data/titanic/titanic_train_knn.csv',
                       test_path='data/titanic/titanic_test_knn.csv'):
    """
    载入并稳健预处理 Titanic 数据集，返回 (X_train, X_test, y_train, y_test)

    处理要点：
    - 合并训练/测试一起清洗以保证特征对齐
    - 删除以 `zero` 开头的冗余列和 Passengerid
    - 处理 `Sex`（兼容字符串或数值）、`Embarked`（独热编码）
    - 数值列用中位数填补，类别列用众数填补
    - 在训练集上 fit `StandardScaler` 并 transform 双方
    """
    print(f"加载训练集: {train_path}")
    print(f"加载测试集: {test_path}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # 标记并合并，方便做统一处理（保证 one-hot 等列一致）
    df_train['_is_train'] = 1
    df_test['_is_train'] = 0

    # 兼容列名：2urvived -> Survived
    if '2urvived' in df_train.columns or '2urvived' in df_test.columns:
        df_train = df_train.rename(columns={'2urvived': 'Survived'})
        df_test = df_test.rename(columns={'2urvived': 'Survived'})

    df = pd.concat([df_train, df_test], ignore_index=True, sort=False)

    # 丢弃无关列（Passengerid / zero* / Unnamed）
    drop_cols = [c for c in df.columns if str(c).lower().startswith('zero')
                 or str(c).lower().startswith('unnamed')
                 or str(c).lower() == 'passengerid']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # 处理 Sex：兼容 'male'/'female' 或数值 0/1
    if 'Sex' in df.columns:
        # 先替换字符串形式
        df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1, 'Male': 0, 'Female': 1})
        df['Sex'] = pd.to_numeric(df['Sex'], errors='coerce')
        # 用训练集众数填充缺失
        if df['Sex'].isnull().any():
            try:
                mode_val = int(df.loc[df['_is_train'] == 1, 'Sex'].mode().iloc[0])
            except Exception:
                mode_val = int(df['Sex'].mode().iloc[0]) if not df['Sex'].mode().empty else 0
            df['Sex'] = df['Sex'].fillna(mode_val).astype(int)
        else:
            df['Sex'] = df['Sex'].astype(int)

    # 处理 Survived（若存在），保证为整数标签
    if 'Survived' in df.columns:
        df['Survived'] = pd.to_numeric(df['Survived'], errors='coerce').fillna(0).astype(int)

    # 数值列用中位数填充（排除 Survived 与标记列）
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for ex in ['Survived', '_is_train']:
        if ex in num_cols:
            num_cols.remove(ex)
    if num_cols:
        medians = df[num_cols].median()
        df[num_cols] = df[num_cols].fillna(medians)

    # 字符串列用众数填充
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    for c in obj_cols:
        if df[c].isnull().any():
            mode = df[c].mode()
            if not mode.empty:
                df[c] = df[c].fillna(mode.iloc[0])
            else:
                df[c] = df[c].fillna('')

    # Embarked 等小类变量做独热编码（如存在）
    if 'Embarked' in df.columns:
        df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked', dummy_na=False)

    # 拆回训练/测试
    df_train_clean = df[df['_is_train'] == 1].drop(columns=['_is_train']).reset_index(drop=True)
    df_test_clean = df[df['_is_train'] == 0].drop(columns=['_is_train']).reset_index(drop=True)

    # 分离 X / y
    if 'Survived' in df_train_clean.columns:
        y_train = df_train_clean['Survived'].astype(int)
        X_train = df_train_clean.drop(columns=['Survived'])
    else:
        y_train = pd.Series([0] * len(df_train_clean), dtype=int)
        X_train = df_train_clean.copy()

    if 'Survived' in df_test_clean.columns:
        y_test = df_test_clean['Survived'].astype(int)
        X_test = df_test_clean.drop(columns=['Survived'])
    else:
        y_test = pd.Series([0] * len(df_test_clean), dtype=int)
        X_test = df_test_clean.copy()

    # 确保 test 的特征与 train 一致（缺失列补 0，多余列删除）
    for c in X_train.columns:
        if c not in X_test.columns:
            X_test[c] = 0
    extra = [c for c in X_test.columns if c not in X_train.columns]
    if extra:
        X_test = X_test.drop(columns=extra)
    # 保持列顺序一致
    X_test = X_test[X_train.columns]

    # 强制数值化（任何残留的非数值列转为数值，缺失填 0）
    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 标准化：在训练集上 fit
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("数据加载与预处理完成。")
    return X_train_scaled, X_test_scaled, y_train, y_test