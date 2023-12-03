import pandas as pd


def load_data():
    """
    Load the dataset
    :return: dataframe
    """
    file_path = 'your_file_path'
    data = pd.read_excel(file_path, sheet_name='Sheet1')  
    
     
    column_names = {
    'brandName': 'brandName',
    'ID': 'ID',
    'content': 'content',
    'cited_brand_number': 'cited_brand_number',
    'cited_brand_name': 'cited_brand_name',
    'rejection_result': 'rejection_result',
    'legal_basis': 'legal_basis',
    'rejected_goods': 'rejected_goods',
    'passed_goods': 'passed_goods',
    '评审文书号': 'review_document_number',
    '申请人': 'applicant',
    '委托代理人': 'authorized_agent',
    '申请人复审的主要理由': 'main_reason_for_review_by_applicant',
    '申请人在复审程序中提交': 'submissions_in_review_procedure',
    '经审理查明': 'findings_after_review',
    '使用法条': 'legal_articles_used',
    '驳回复审结果': 'review_rejection_result',
    '裁定日期': 'ruling_date'
}

    # Rename the columns in the dataframe
    data = data.rename(columns=column_names)
    
    return data
def clean_data(data):
    """
    Clean the dataset
    :param data: dataframe
    :return: cleaned dataframe
    """
    # Drop the rows with missing values
    data = data.dropna()
    
    data = data.drop_duplicates()
    
    data = data.reset_index(drop=True)
    
    return data
def clean_data(data):
    """
    Clean the dataset
    Check for missing values in the dataset
    delete all rows with missing values
    delete duplicate rows
    reset the index of the dataframe
    :param data: dataframe
    :return: cleaned dataframe
    """
    missing_values = data.isnull().sum()
    missing_values_percentage = (missing_values / len(data)) * 100
    missing_data = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_values_percentage
    })
    missing_data.sort_values(by='Percentage', ascending=False)
    # 1. Drop columns with more than 90% missing values
    cols_to_drop = missing_data[missing_data['Percentage'] > 90].index.tolist()
    data_cleaned = data.drop(columns=cols_to_drop)

    # 2. Fill missing values
    # For simplicity, we are treating all columns as non-numeric. So, we will fill them with '未知'
    data_cleaned.fillna('-1', inplace=True)

    # 3. Drop potential data leakage columns
    # As mentioned, the exact names of these columns in the provided dataset aren't clear due to encoding.
    # We will drop the column "legal_basis" (which seems to correspond to one of the mentioned columns).
    data_cleaned = data_cleaned.drop(columns=['legal_basis'])
    data_cleaned = data_cleaned[data_cleaned['rejection_result'] != '-1']

    return data_cleaned
                                
                                
def feature_engineering_new_features(data_cleaned):
    """
    Perform feature engineering on the dataset
    :param data: cleaned dataframe
    :return: dataframe with new features
    """
    # Create a new column called "similarity_scores" which contains the concatenation of the brand_name and cited_brand_name columns
    # This is done to make the model more robust to different brand names
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Combine the two columns for TF-IDF vectorization
    combined_texts = data_cleaned['brandName'].astype(str) + data_cleaned['cited_brand_name'].astype(str)

    # Vectorize the combined texts
    vectorizer = TfidfVectorizer()
    # Fit the vectorizer on combined texts
    vectorizer.fit(combined_texts)

    # Transform the brand_name and cited_name columns individually using the fitted vectorizer
    brandName_matrix = vectorizer.transform(data_cleaned['brandName'].astype(str))
    cited_brand_name_matrix = vectorizer.transform(data_cleaned['cited_brand_name'].astype(str))

    brandName_matrix.shape, cited_brand_name_matrix.shape

    # Compute the cosine similarity between the two sets of vectors
    similarities = cosine_similarity(brandName_matrix, cited_brand_name_matrix)
    # Since cosine_similarity returns a matrix, we take the diagonal which gives the similarity of each pair
    similarity_scores = similarities.diagonal()

    # Add the similarity scores to the dataframe as a new column
    data_cleaned['similarity_score'] = similarity_scores
    
    # # Display the first few rows of the data with the new column
    # data_cleaned[['similarity_score','brandName', 'cited_brand_name' ]].head()

    # Set the similarity_score to NaN for rows where similarity is 0
    data_cleaned.loc[data_cleaned['similarity_score'] == 0, 'similarity_score'] = np.nan

    # Fill NaN values in similarity_score with its mean，median，mode_similarity
    # mean_similarity = data_cleaned['similarity_score'].mean()
    # data_cleaned['similarity_score'].fillna(mean_similarity, inplace=True)

    # median_similarity = data_cleaned['similarity_score'].median()
    # data_cleaned['similarity_score'].fillna(median_similarity, inplace=True)

    mode_similarity = data_cleaned['similarity_score'].mode()[0]  # mode() 返回一个Series，因此我们选择第一个值
    data_cleaned['similarity_score'].fillna(mode_similarity, inplace=True)

    return data_cleaned


def feature_engineering_categorization(data_cleaned):
    mapping = {
    '予以驳回': 0,
    '予以初步审定': 1,
    '部分驳回': 2
}
    data_cleaned['rejection_result'] = data_cleaned['rejection_result'].map(mapping)
    return data_cleaned



