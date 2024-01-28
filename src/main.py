import pandas as pd
from xgboost import XGBClassifier
import feature_engineering as feature_engineering
from mi_analysis import MiScores
from data_preprocessor import DataPreprocessor
from model_manager import ModelManager
from model_eval import ModelEvaluator

def main():
    try:
        # 数据加载和清洗
        data = feature_engineering.load_data()
        cleaned_data = feature_engineering.clean_data(data)

        # 特征工程
        data_with_new_features = feature_engineering.feature_engineering_new_features(cleaned_data)
        final_data = feature_engineering.feature_engineering_categorization(data_with_new_features)

        # 目标列和特征列定义
        target_column = 'rejection_result'
        selected_features = ['similarity_score', 'brandName', 'cited_brand_name', 'authorized_agent', 'main_reason_for_review_by_applicant']

        # 互信息评分
        mi_analysis = MiScores(final_data)
        mi_scores = mi_analysis.compute_and_plot_mi(selected_features, target_column, discrete_features=(final_data.dtypes == int))
        print(mi_scores)

        # 数据预处理
        preprocessor = DataPreprocessor(final_data, selected_features)
        preprocessed_data = preprocessor.preprocess()
        numeric_cols = [cname for cname in preprocessed_data.columns if preprocessed_data[cname].dtype in ['int64', 'float64']]
        X_train, X_valid, X_test, y_train, y_valid, y_test = preprocessor.split_data(target_column)

        # 模型训练和评估
        manager = ModelManager(X_train, y_train, X_valid, y_valid, X_test, y_test, numeric_cols)
        validation_score, accuracy_valid, report_valid, accuracy_test, report_test = manager.train_and_evaluate()
        print("Validation Score:", validation_score)
        print("Validation Accuracy:", accuracy_valid)
        print("Validation Report:\n", report_valid)
        print("Test Accuracy:", accuracy_test)
        print("Test Report:\n", report_test)

        # 网格搜索
        param_grid = {
            'model__n_estimators': [100*i for i in range(1,11)],
            'model__learning_rate': [0.01*i for i in range(1,10)],
        }
        best_params = manager.perform_grid_search(param_grid)
        print("Best Parameters:", best_params)

        manager.save_model('xgboost_classification_model_gridsearchcv.json')

        # 模型评估
        model = XGBClassifier()
        model.load_model('xgboost_classification_model_gridsearchcv.json')
        evaluator = ModelEvaluator(model, X_test, y_test, selected_features)
        evaluator.plot_tree('./tree_gridsearchcv.png')
        evaluator.plot_feature_importance()
        evaluator.plot_confusion_matrix()

        y_pred = model.predict(X_test)
        evaluator.plot_performance_metrics(y_pred)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        evaluator.plot_roc_curve(y_pred_proba)
        accuracy, precision, recall, f1 = evaluator.evaluate_performance()
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
