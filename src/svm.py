import os
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_svm(X_train, y_train, model_save_path="models/svm_titanic_20260423.pkl"):
    """
    训练 SVM 模型并保存
    """
    print("开始训练 SVM 模型...")
    # 这里默认使用 rbf 核函数，你也可以尝试 kernel='linear' 观察准确率变化
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    model.fit(X_train, y_train)
    
    # 确保 models 文件夹存在
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # 保存模型参数
    joblib.dump(model, model_save_path)
    print(f"✅ 模型已成功保存至: {model_save_path}")
    
    return model

def test_svm(X_test, y_test, model_path="models/svm_titanic_20260423.pkl"):
    """
    加载训练好的 SVM 模型并进行测试，输出准确率
    """
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}，请先执行训练！")
        return
        
    print(f"正在加载模型: {model_path} ...")
    model = joblib.load(model_path)
    
    # 进行预测
    predictions = model.predict(X_test)
    
    # 计算并输出准确率
    accuracy = accuracy_score(y_test, predictions)
    print(f"📊 SVM 模型在测试集上的准确率为: {accuracy * 100:.2f}%")
    
    # 可以将结果保存到 results/ 文件夹中
    os.makedirs("results", exist_ok=True)
    with open("results/accuracy.txt", "a", encoding="utf-8") as f:
        f.write(f"SVM_Titanic_Accuracy: {accuracy * 100:.2f}%\n")
        
    return accuracy