# Credit Card Fraud Detector

### Problem: 
I am tasked with developing a robust credit card fraud detection system capable of accurately identifying fraudulent transactions in real-time. Given the highly unbalanced nature of the dataset, with only a small percentage of transactions being fraudulent, my goal is to implement effective machine learning algorithms that can handle class imbalance and provide high precision and recall rates. By achieving this, credit card companies can minimize financial losses due to fraud while ensuring a smooth and secure experience for cardholders.

## Project Topic
This project aims to develop an effective credit card fraud detection system utilizing machine learning algorithms to accurately identify fraudulent transactions in real-time. Given the imbalanced nature of the dataset, with only a small percentage of transactions being fraudulent, the goal is to implement models capable of handling class imbalance and achieving high precision and recall rates. Key techniques include Isolation Forest and KMeans clustering, with evaluation metrics such as AUC score and F1 score used to assess model performance.

#### Description:
The dataset consists of credit card transactions made by European cardholders in September 2013. It includes features such as the time elapsed since the first transaction, anonymized numerical features resulting from PCA dimensionality reduction (V1-V28), transaction amount, and a binary label indicating whether the transaction is fraudulent or genuine. Due to privacy concerns, the original features and additional background information are not provided. However, the dataset's highly unbalanced nature with only a small percentage of fraudulent transactions (0.17%) presents a challenging scenario for fraud detection algorithms.

#### Potential Applications:
Financial Institutions: Improve credit card fraud detection systems to minimize financial losses and enhance security for cardholders.

Machine Learning Researchers: Develop and evaluate algorithms for handling imbalanced datasets and detecting fraudulent activities in real-time.

Regulatory Authorities: Gain insights into the prevalence and characteristics of credit card fraud to inform policy-making and regulatory measures aimed at preventing financial crimes.

## Data

The dataset, named "Credit Card Fraud Detection," was obtained from Kaggle and curated by the Machine Learning Group of Université Libre de Bruxelles (ULB) in collaboration with Worldline. It contains credit card transaction data from September 2013, comprising both fraudulent and genuine transactions made by European cardholders. The dataset features numerical input variables resulting from PCA transformation, along with original features such as transaction time and amount. Due to privacy concerns, the original features and additional background information are not provided. With a highly unbalanced distribution, fraud cases account for only 0.172% of transactions. This dataset serves as a valuable resource for developing and evaluating machine learning models to detect fraudulent activities in credit card transactions.

Reference
MLG-ULB. (n.d.). Credit Card Fraud Detection [Data set]. Kaggle. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data

#### Data Description
The dataset provided contains credit card transactions made by European cardholders during September 2013, comprising a total of 284,807 transactions. Each transaction is represented as a row in the dataset, with 31 features provided for analysis. The dataset size is approximately 69 megabytes. The features in the dataset are primarily numerical, with 28 features labeled as V1 through V28, representing the principal components obtained through PCA (Principal Component Analysis) transformation. The purpose of PCA is to reduce the dimensionality of the data while preserving important information and protecting sensitive features. Additionally, there are two non-transformed features: "Time" and "Amount."

Features:

- Time: The number of seconds elapsed between the current transaction and the first transaction in the dataset. This feature can provide insights into the temporal patterns of transactions.
- V1-V28: Principal components obtained through a PCA transformation, anonymized to protect user identities and sensitive features.
- Amount: The monetary value of the transaction.
- Class: A binary label indicating whether the transaction is fraudulent (1) or genuine (0).
The dataset primarily consists of numerical input variables, with only two non-transformed features: 'Time' and 'Amount'. 'Time' represents the time elapsed since the first transaction, while 'Amount' denotes the transaction amount. The remaining features, V1 to V28, are principal components obtained through PCA transformation.

The dataset exhibits a highly unbalanced class distribution, with fraudulent transactions accounting for only 0.172% of all transactions. This imbalance poses a challenge for traditional classification algorithms, as accurate detection of fraudulent transactions requires specialized techniques to handle class imbalance effectively.

Given the imbalance ratio, it is recommended to evaluate model performance using metrics such as the Area Under the Precision-Recall Curve (AUPRC), which provides a more informative assessment of classifier performance for imbalanced datasets compared to traditional accuracy measures.




## Data Cleaning

In the data cleaning process, I performed several steps to ensure the quality and integrity of the dataset. Here's a summary of the data cleaning steps:

- Removal of duplicate rows: I identified and removed duplicate rows from the dataset to prevent redundancy and ensure each transaction is unique.

- Handling missing values: I checked for any missing values in the dataset and confirmed that there were none, ensuring that all data points were complete.

- Outlier detection and removal: I used z-score method to detect outliers in the 'Amount' column and removed them from the dataset. This helped to improve the accuracy of the analysis by eliminating potentially erroneous data points.

- Data type validation: I validated the data types of all columns in the dataset to ensure consistency and compatibility with analysis techniques. All columns were confirmed to have the appropriate data types for further analysis.

- Verification of class balance: I examined the distribution of the target variable 'Class' to ensure that the dataset was not heavily skewed towards one class. The dataset was found to be highly unbalanced, with fraudulent transactions accounting for only 0.17% of all transactions.

Overall, these data cleaning steps helped to prepare the dataset for further analysis and modeling, ensuring that it was free from duplicates, outliers, missing values, and had consistent data types.

### Exploratory Data Analysis

The exploratory data analysis (EDA) for the credit card fraud detection dataset begins with an examination of the distribution of numerical features. Histograms are utilized to visualize the distributions of features such as 'Time,' 'Amount,' and the anonymized features labeled as 'V1' through 'V28.' The histograms provide insights into the distribution patterns of these features, allowing for a better understanding of their characteristics. Additionally, the histograms help in identifying potential outliers and understanding the range and spread of values within each feature.

Following the examination of individual feature distributions, a correlation matrix is generated to explore relationships between the anonymized features and the target variable 'Class' (indicating fraudulent or non-fraudulent transactions). The heatmap visualization of the correlation matrix enables the identification of any significant correlations between the features and the target variable. This analysis helps in identifying which features may have a stronger influence on the classification of transactions as fraudulent or non-fraudulent.

Moreover, specific attention is given to the distribution of transactions over time, with a focus on identifying any patterns or trends that may exist. Kernel Density Estimation (KDE) plots are used to visualize the density of credit card transactions over time, comparing the distributions between fraudulent and non-fraudulent transactions. This analysis aids in understanding if there are specific time periods or trends associated with fraudulent activities.

Furthermore, the analysis includes visualization of the dollar amounts of transactions for both fraudulent and non-fraudulent cases. Histograms are employed to display the distribution of transaction amounts, with the y-axis scaled logarithmically for better visualization. This allows for a comparison of the frequency of transactions across different dollar amount ranges, highlighting any differences between fraudulent and non-fraudulent transactions in terms of transaction amounts.

In summary, the exploratory data analysis provides a comprehensive overview of the dataset, examining the distributions of numerical features, exploring correlations with the target variable, analyzing transaction patterns over time, and comparing transaction amounts between fraudulent and non-fraudulent cases. Through visualizations and analyses, the EDA offers valuable insights into the characteristics and potential factors influencing credit card fraud detection.

### Models

In the context of credit card fraud detection, the selection of appropriate models is crucial for effectively identifying fraudulent transactions while minimizing false positives. For this task, unsupervised learning models are often utilized due to the lack of labeled fraudulent transactions for training. Two common unsupervised learning models employed in credit card fraud detection are Isolation Forest and KMeans clustering.

Isolation Forest is a tree-based anomaly detection algorithm that isolates outliers in the data by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. This process is repeated recursively until all data points are isolated. Anomalies are identified as data points that require fewer splits to isolate, indicating that they are different from the majority of the data.

KMeans clustering is a partition-based clustering algorithm that partitions data into K clusters based on similarity. It works by iteratively assigning data points to the nearest cluster centroid and then updating the centroids based on the mean of the data points assigned to each cluster. Anomalies can be identified as data points that do not belong to any of the clusters or are distant from their cluster centroids.

Both Isolation Forest and KMeans clustering offer advantages and limitations in the context of credit card fraud detection. Isolation Forest is effective at isolating individual fraudulent transactions as anomalies, making it suitable for detecting outliers in high-dimensional data. However, it may struggle with identifying fraud patterns that span multiple transactions or exhibit complex relationships between features. On the other hand, KMeans clustering can capture clusters of fraudulent transactions based on similarity, allowing for the detection of fraud patterns that involve multiple transactions. However, it may be sensitive to the choice of the number of clusters (K) and the initial centroid positions, and it assumes that clusters are spherical and have similar sizes.

In practice, a combination of Isolation Forest and KMeans clustering, along with other techniques such as feature engineering and ensemble methods, may be used to improve the accuracy and robustness of credit card fraud detection systems. Additionally, model evaluation metrics such as AUC-ROC and F1 score are commonly used to assess the performance of unsupervised learning models in detecting fraudulent transactions.

#### Model Building and Evaluation

##### Model Comparison

- Isolation Forest:
  - AUC Score : 0.790846525589921
  - F1 Score : 0.15885947046843177

- KMeans:
  - AUC Score (KMeans): 0.5242803269964921
  - F1 Score (KMeans): 0.006008782066096603


## Result and Analysis

In the evaluation of unsupervised models for credit card fraud detection, two primary algorithms were employed: Isolation Forest and K-Means clustering. These models were selected due to their ability to identify anomalies or clusters within the data without requiring labeled examples of fraud. The performance of each model was assessed using metrics such as Area Under the Curve (AUC) for Receiver Operating Characteristic (ROC) and F1-score.

The Isolation Forest model demonstrated superior performance compared to K-Means clustering. With an AUC score of approximately 0.79 and an F1-score of around 0.16, the Isolation Forest model outperformed K-Means clustering, which had an AUC score of approximately 0.52 and an F1-score of only about 0.006. These results indicate that the Isolation Forest model was more effective at distinguishing between normal and fraudulent transactions.

One possible explanation for the better performance of the Isolation Forest model is its ability to isolate anomalies by constructing isolation trees. This approach is particularly well-suited for identifying rare events, such as fraudulent transactions, which may be outliers in the dataset. In contrast, K-Means clustering may struggle to effectively separate fraudulent transactions from normal ones, as it relies on grouping data points into clusters based on similarity.

Furthermore, the Isolation Forest model demonstrated a higher level of robustness and generalization, as evidenced by its higher F1-score. This suggests that the model was better able to accurately classify both fraudulent and non-fraudulent transactions, reducing the likelihood of false positives and false negatives.

In conclusion, the Isolation Forest model proved to be a more effective and reliable approach for credit card fraud detection in this scenario. Its ability to isolate anomalies and its robust performance metrics make it a valuable tool for financial institutions seeking to mitigate the risks associated with fraudulent transactions. However, further research and experimentation with different algorithms and techniques may still be warranted to continuously improve fraud detection systems and stay ahead of evolving fraud tactics.

## Discussion and Conclusion

The exploration into credit card fraud detection using machine learning models has yielded valuable insights and considerations for effectively identifying fraudulent transactions. The dataset provided a challenging scenario due to its highly imbalanced nature, with fraudulent transactions accounting for only a small fraction of the total.

Two primary models were evaluated for their effectiveness in detecting fraudulent activity: Isolation Forest and KMeans clustering. Both models were trained on a subset of features derived from the anonymized credit card transaction data, which included time, amount, and principal components obtained through PCA transformation.

The Isolation Forest model demonstrated better performance compared to KMeans clustering, as evidenced by its higher AUC score and F1 score. The Isolation Forest algorithm is well-suited for anomaly detection tasks, making it particularly effective for identifying rare instances of fraudulent transactions within the dataset. On the other hand, KMeans clustering struggled to distinguish between normal and fraudulent transactions, resulting in lower performance metrics.

One key takeaway from this analysis is the importance of choosing appropriate algorithms for anomaly detection tasks, especially in highly imbalanced datasets. The Isolation Forest model's ability to isolate anomalies by constructing random decision trees and identifying instances that are isolated in fewer steps proved to be effective for detecting fraudulent transactions.

Furthermore, the evaluation of feature importance and correlation revealed insights into the underlying patterns and relationships within the data. While the dataset's anonymized nature limited the interpretability of individual features, the overall correlation analysis provided valuable context for understanding the data's structure.

In conclusion, this analysis underscores the importance of selecting suitable machine learning algorithms and feature engineering techniques for credit card fraud detection. By leveraging advanced anomaly detection methods like Isolation Forest and conducting thorough evaluations, financial institutions can enhance their ability to identify and prevent fraudulent transactions, thereby safeguarding both customers and businesses from financial losses. Additionally, this analysis highlights the need for ongoing research and development in fraud detection methodologies to stay ahead of evolving fraudulent tactics and ensure the security of financial transactions.

#### Recommendations

- Artists and producers can leverage the insights from this model to optimize their songs for Spotify success.
- Focus on features with higher importance, as identified by the Random Forest model.
- Regularly update the model with new data to adapt to changing music trends.

Feel free to explore the Jupyter notebook for a detailed walkthrough of the analysis and modeling process.
