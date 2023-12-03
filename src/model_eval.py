
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import plot_tree, plot_importance

class ModelEvaluator:
    def __init__(self, model, X_test, y_test, selected_features):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.selected_features = selected_features

    def plot_tree(self, filename, dpi=300):
        plt.figure(figsize=(25, 25), dpi=dpi)
        plot_tree(self.model, num_trees=0, rankdir='LR')
        plt.savefig(filename, format='png', dpi=dpi)
        plt.show()

    def plot_feature_importance(self):
        feature_importance = self.model.feature_importances_
        print("Length of feature_importance:", len(feature_importance))
        print("Length of X.columns:", len(X.columns))
        print(X.columns)

        # Assuming selected_features is the list of original features
        feature_groups = {feature: [col for col in X.columns if col.startswith(feature)] for feature in selected_features}
        average_importance = {}

        for feature, columns in feature_groups.items():
            indices = [X.columns.get_loc(col) for col in columns if X.columns.get_loc(col) < len(feature_importance)]
            if indices:  # Ensure there are valid indices
                avg_importance = feature_importance[indices].mean()
                average_importance[feature] = avg_importance
        print(average_importance)
        # Getting the sorted features based on their average importance
        sorted_features = sorted(average_importance, key=average_importance.get, reverse=True)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_features, [average_importance[feature] for feature in sorted_features])
        plt.xlabel("Average Importance")
        plt.ylabel("Original Feature")
        plt.title("Average Feature Importance")
        plt.tight_layout()
        plt.show()

        # 指定要绘制的特征顺序
        ordered_features = ['similarity_score', 'brandName', 'cited_brand_name',  
                            'authorized_agent', 'main_reason_for_review_by_applicant']

        # 计算各特征组平均重要性  
        feature_groups = {feature: [col for col in X.columns if col.startswith(feature)] for feature in selected_features}
        average_importance = {}
        for feature, columns in feature_groups.items():
            indices = [X.columns.get_loc(col) for col in columns if X.columns.get_loc(col) < len(feature_importance)]
            if indices:
                avg_importance = feature_importance[indices].mean()
                average_importance[feature] = avg_importance
                
        # 按指定顺序取出重要性值        
        ordered_importances = [average_importance[feature] for feature in ordered_features]

        # 绘制柱状图
        plt.figure(figsize=(10, 6)) 
        plt.barh(ordered_features, ordered_importances)
        plt.xlabel("Average Importance")
        # plt.ylabel("Original Feature")
        plt.title("Average Feature Importance")
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self):
        conf_mat = confusion_matrix(self.y_test, self.model.predict(self.X_test))
        sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    def evaluate_performance(self):
        y_pred = self.model.predict(self.X_test)
        report = classification_report(self.y_test, y_pred)
        print("Classification Report:\n", report)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='macro')
        recall = recall_score(self.y_test, y_pred, average='macro')
        f1 = f1_score(self.y_test, y_pred, average='macro')

        return accuracy, precision, recall, f1
    def plot_performance_metrics(self, y_pred):
        # 生成分类报告
        report = classification_report(self.y_test, y_pred, output_dict=True)
        metrics_df = pd.DataFrame(report).transpose()

        # 选择所需指标
        selected_metrics = metrics_df.loc[:, ['precision', 'recall', 'f1-score']]

        # 绘制条形图
        selected_metrics.plot(kind='bar', figsize=(10, 6))
        plt.title("Accuracy, Recall and F1 Score Comparison")
        plt.ylabel("Score")
        plt.xlabel("Classes")
        plt.show()

    def plot_grid_search_results(self, grid_search):
        # 从grid对象中提取结果，并将其转换为DataFrame
        results = pd.DataFrame(grid_search.cv_results_)

        # 创建热图
        pivot = results.pivot(index='param_n_estimators', columns='param_learning_rate', values='mean_test_score')
        plt.figure(figsize=(14, 10))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap='viridis')
        plt.xlabel('Learning Rate')
        plt.ylabel('Number of Estimators')
        plt.title('Mean Test Score for different combinations of n_estimators and learning_rate')
        plt.show()

    def plot_roc_curve(self, y_pred_proba):
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label='XGBoost Model (AUC: %.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
