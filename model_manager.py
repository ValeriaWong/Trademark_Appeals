import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, make_scorer
from sklearn.model_selection import cross_val_score, GridSearchCV

class ModelManager:
    def __init__(self, X_train, y_train, X_valid, y_valid, X_test, y_test, numeric_cols):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test
        self.numeric_cols = numeric_cols
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='constant'), self.numeric_cols),
            ])
        self.model = XGBClassifier(n_jobs=-1, tree_method='gpu_hist', gpu_id=0)
        self.pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                        ('model', self.model)])
        self.scorer = make_scorer(accuracy_score)

    def train_and_evaluate(self):
        # Train
        self.pipeline.fit(self.X_train, self.y_train)

        # Validate
        validation_score = self.pipeline.score(self.X_valid, self.y_valid)
        y_pred_valid = self.pipeline.predict(self.X_valid)
        accuracy_valid = accuracy_score(self.y_valid, y_pred_valid)
        report_valid = classification_report(self.y_valid, y_pred_valid)

        # Test
        y_pred_test = self.pipeline.predict(self.X_test)
        accuracy_test = accuracy_score(self.y_test, y_pred_test)
        report_test = classification_report(self.y_test, y_pred_test)

        return validation_score, accuracy_valid, report_valid, accuracy_test, report_test

    def perform_grid_search(self, param_grid):
        grid_search = GridSearchCV(self.pipeline, param_grid, cv=5, scoring=self.scorer)
        grid_search.fit(self.X_train, self.y_train)
        return grid_search.best_params_

    def save_model(self, file_path):
        self.pipeline.named_steps['model'].save_model(file_path)

    def load_model(self, file_path):
        self.model.load_model(file_path)
        self.pipeline.named_steps['model'] = self.model

    def predict_and_save(self, file_path):
        y_pred = self.pipeline.predict(self.X_test)
        output = pd.DataFrame({'Id': self.X_test.index, 'Decision_pred': y_pred})
        output.to_csv(file_path, index=False)
