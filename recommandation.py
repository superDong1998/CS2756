import gzip
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import sys
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def parse(filename):
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        entry = {}
        for l in f:
            l = l.strip()
            colonPos = l.find(':')
            if colonPos == -1:
                yield entry
                entry = {}
                continue
            eName = l[:colonPos]
            rest = l[colonPos + 2:]
            entry[eName] = rest
        yield entry

def load_data(filename):
    data = []
    for e in parse(filename):
        if "product/productId" in e and "review/userId" in e and "review/score" in e and "product/title" in e and "review/text" in e:
            e['sentiment'] = sia.polarity_scores(e['review/text'])['compound']
            data.append(e)
    return pd.DataFrame(data)

def get_actual_likes(df):
    df['review/score'] = df['review/score'].astype(float)
    return df[df['review/score'] >= 4].groupby('review/userId')['product/title'].apply(set).to_dict()

def evaluate_recommendations(recommendations, actual_likes):
    precision_list, recall_list, f1_list = [], [], []
    
    for user, recommended_items in recommendations.items():
        actual_items = actual_likes.get(user, set())
        recommended_set = set(recommended_items)

        if not actual_items:
            continue

        true_positives = len(recommended_set & actual_items)
        precision = true_positives / len(recommended_set) if recommended_set else 0
        recall = true_positives / len(actual_items) if actual_items else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "Precision": np.mean(precision_list),
        "Recall": np.mean(recall_list),
        "F1-score": np.mean(f1_list)
    }

def kmeans_recommend(df, n_clusters=5, top_k=5):
    pivot_table = df.pivot_table(index='review/userId', columns='product/title', values='review/score', fill_value=0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(pivot_table)

    user_cluster_map = pd.DataFrame({'review/userId': pivot_table.index, 'cluster': clusters})
    df = df.merge(user_cluster_map, on='review/userId', how='left')

    recommendations = {}
    for cluster in range(n_clusters):
        cluster_users = df[df['cluster'] == cluster]['review/userId'].unique()
        cluster_products = df[df['cluster'] == cluster]['product/title'].value_counts().index[:top_k].tolist()
        for user in cluster_users:
            recommendations[user] = cluster_products

    return recommendations

def collaborative_filtering(df, sentiment, top_k=5):
    df_filtered = df[df['sentiment'] > 0] if sentiment else df
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df_filtered[['review/userId', 'product/title', 'review/score']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    model = KNNBasic(sim_options={'user_based': True})
    model.fit(trainset)
    predictions = model.test(testset)

    recommendations = defaultdict(list)
    for uid, iid, _, est, _ in predictions:
        recommendations[uid].append((iid, est))

    for uid in recommendations:
        recommendations[uid] = [title for title, _ in sorted(recommendations[uid], key=lambda x: x[1], reverse=True)[:top_k]]

    return recommendations

def bayesian_recommend(df, top_k=5):
    product_ratings = defaultdict(list)
    for _, row in df.iterrows():
        product_ratings[row['product/title']].append(float(row['review/score']))
    bayes_scores = {prod: np.mean(ratings) for prod, ratings in product_ratings.items()}

    return [prod for prod, _ in sorted(bayes_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]

def train_model_based(df, model_type='rf', top_k=5, use_sentiment=False, user_sample=None):
    le_user = LabelEncoder()
    le_product = LabelEncoder()
    gpu = False

    df['user_id_enc'] = le_user.fit_transform(df['review/userId'])
    df['product_id_enc'] = le_product.fit_transform(df['product/title'])

    features = ['user_id_enc', 'product_id_enc']
    if use_sentiment:
        features.append('sentiment')

    X = df[features]
    y = df['review/score'].astype(float)

    if model_type == 'rf':
        model = RandomizedSearchCV(RandomForestRegressor(), {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None]
        }, n_iter=5, cv=3, n_jobs=-1, random_state=42)
    elif model_type == 'xgb':
        if gpu:
            model = RandomizedSearchCV(XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor'), {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.3]
            }, n_iter=5, cv=3, n_jobs=-1, random_state=42)
        else:
            model = RandomizedSearchCV(XGBRegressor(), {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.3]
            }, n_iter=5, cv=3, n_jobs=-1, random_state=42)

    model.fit(X, y)

    all_users = df['review/userId'].unique()
    all_products = df['product/title'].unique()
    if user_sample:
        all_users = all_users[:user_sample]
        # all_products = all_products[:100]  # Limit products

    user_avg_sentiment = df.groupby('review/userId')['sentiment'].mean().to_dict() if use_sentiment else {}

    recommendations = defaultdict(list)
    for user in all_users:
        user_df = pd.DataFrame({
            'review/userId': [user] * len(all_products),
            'product/title': all_products
        })
        user_df['user_id_enc'] = le_user.transform([user] * len(all_products))
        user_df['product_id_enc'] = le_product.transform(all_products)

        if use_sentiment:
            sentiment_val = user_avg_sentiment.get(user, 0.0)
            user_df['sentiment'] = sentiment_val

        X_test = user_df[features]
        preds = model.predict(X_test)
        user_df['pred_score'] = preds
        top_items = user_df.sort_values(by='pred_score', ascending=False)['product/title'].head(top_k).tolist()
        recommendations[user] = top_items

    return recommendations

def usage():
    print("Welcome to recommendation system!")
    print("Please use following command format:")
    print("     python recommendation.py your_data_set top_k")
    print()
    exit(0)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        usage()

    filename = sys.argv[1]
    top_k = int(sys.argv[2])

    df = load_data(filename)
    actual_likes = get_actual_likes(df)

    kmeans_recommendations = kmeans_recommend(df, top_k=top_k)
    kmeans_metrics = evaluate_recommendations(kmeans_recommendations, actual_likes)
    print("K-Means Clustering Done.")
    print("K-Means Metrics:", kmeans_metrics)

    cf_recommendations = collaborative_filtering(df, sentiment=False, top_k=top_k)
    cf_with_sentiment = collaborative_filtering(df, sentiment=True, top_k=top_k)
    cf_metrics = evaluate_recommendations(cf_recommendations, actual_likes)
    cf_metrics_sentiment = evaluate_recommendations(cf_with_sentiment, actual_likes)
    print("Collaborative Filtering Done.")
    print("Collaborative Filtering Metrics:", cf_metrics)
    print("Collaborative Filtering with Sentiment Metrics:", cf_metrics_sentiment)

    bayes_recommendations = bayesian_recommend(df, top_k=top_k)
    bayes_user_based = {user: bayes_recommendations for user in df['review/userId'].unique()}
    bayes_metrics = evaluate_recommendations(bayes_user_based, actual_likes)
    print("Bayesian Recommendation Done.")
    print("Top Bayesian Recommendations:", bayes_recommendations)
    print("Bayesian Recommendation Metrics:", bayes_metrics)

    rf_recommendations = train_model_based(df, model_type='rf', top_k=top_k, use_sentiment=False, user_sample=None)
    rf_metrics = evaluate_recommendations(rf_recommendations, actual_likes)
    print("Random Forest Recommendation Done.")
    print("Random Forest Metrics:", rf_metrics)

    rf_sentiment_recommendations = train_model_based(df, model_type='rf', top_k=top_k, use_sentiment=True, user_sample=None)
    rf_sentiment_metrics = evaluate_recommendations(rf_sentiment_recommendations, actual_likes)
    print("Random Forest with Sentiment Recommendation Done.")
    print("Random Forest with Sentiment Metrics:", rf_sentiment_metrics)

    xgb_recommendations = train_model_based(df, model_type='xgb', top_k=top_k, use_sentiment=False, user_sample=None)
    xgb_metrics = evaluate_recommendations(xgb_recommendations, actual_likes)
    print("XGBoost Recommendation Done.")
    print("XGBoost Metrics:", xgb_metrics)

    xgb_sentiment_recommendations = train_model_based(df, model_type='xgb', top_k=top_k, use_sentiment=True, user_sample=None)
    xgb_sentiment_metrics = evaluate_recommendations(xgb_sentiment_recommendations, actual_likes)
    print("XGBoost with Sentiment Recommendation Done.")
    print("XGBoost with Sentiment Metrics:", xgb_sentiment_metrics)

    # Visualization of metrics
    algorithms = [
        "KMeans", "CF", "CF+Senti", "Bayes", 
        "RF", "RF+Senti", "XGB", "XGB+Senti"
    ]
    precision_scores = [
        kmeans_metrics['Precision'], cf_metrics['Precision'], cf_metrics_sentiment['Precision'],
        bayes_metrics['Precision'], rf_metrics['Precision'], rf_sentiment_metrics['Precision'],
        xgb_metrics['Precision'], xgb_sentiment_metrics['Precision']
    ]
    recall_scores = [
        kmeans_metrics['Recall'], cf_metrics['Recall'], cf_metrics_sentiment['Recall'],
        bayes_metrics['Recall'], rf_metrics['Recall'], rf_sentiment_metrics['Recall'],
        xgb_metrics['Recall'], xgb_sentiment_metrics['Recall']
    ]
    f1_scores = [
        kmeans_metrics['F1-score'], cf_metrics['F1-score'], cf_metrics_sentiment['F1-score'],
        bayes_metrics['F1-score'], rf_metrics['F1-score'], rf_sentiment_metrics['F1-score'],
        xgb_metrics['F1-score'], xgb_sentiment_metrics['F1-score']
    ]

    x = np.arange(len(algorithms))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precision_scores, width, label='Precision')
    plt.bar(x, recall_scores, width, label='Recall')
    plt.bar(x + width, f1_scores, width, label='F1-Score')

    plt.xticks(x, algorithms, rotation=45)
    plt.ylabel("Score")
    plt.set_yscale('log')
    plt.title("Metrics by Algorithm")
    plt.legend()
    plt.savefig(f"{filename.split('/')[-1].replace('.gz','')}_top{top_k}.pdf", format='pdf')