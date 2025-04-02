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
            continue  # Skip users with no actual likes
        
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

def kmeans_recommend(df, n_clusters=5):
    pivot_table = df.pivot_table(index='review/userId', columns='product/title', values='review/score', fill_value=0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(pivot_table)
    
    user_cluster_map = pd.DataFrame({'review/userId': pivot_table.index, 'cluster': clusters})
    df = df.merge(user_cluster_map, on='review/userId', how='left')
    
    recommendations = {}
    for cluster in range(n_clusters):
        cluster_users = df[df['cluster'] == cluster]['review/userId'].unique()
        cluster_products = df[df['cluster'] == cluster]['product/title'].value_counts().index[:5].tolist()
        for user in cluster_users:
            recommendations[user] = cluster_products
    
    return recommendations

def collaborative_filtering(df, sentiment):
    df_filtered = df[df['sentiment'] > 0]  # Remove negative sentiment reviews
    reader = Reader(rating_scale=(0, 5))
    if sentiment:
        data = Dataset.load_from_df(df_filtered[['review/userId', 'product/title', 'review/score']], reader)
    else:
        data = Dataset.load_from_df(df[['review/userId', 'product/title', 'review/score']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    model = KNNBasic(sim_options={'user_based': True})
    model.fit(trainset)
    predictions = model.test(testset)
    
    recommendations = defaultdict(list)
    for uid, iid, _, est, _ in predictions:
        recommendations[uid].append((iid, est))
    
    for uid in recommendations:
        recommendations[uid] = [title for title, _ in sorted(recommendations[uid], key=lambda x: x[1], reverse=True)[:5]]
    
    return recommendations

def bayesian_recommend(df):
    product_ratings = defaultdict(list)
    for _, row in df.iterrows():
        product_ratings[row['product/title']].append(float(row['review/score']))
    bayes_scores = {prod: np.mean(ratings) for prod, ratings in product_ratings.items()}
    
    return [prod for prod, _ in sorted(bayes_scores.items(), key=lambda x: x[1], reverse=True)[:5]]

if __name__ == "__main__":
    filename = "dataset/Arts.txt.gz"
    df = load_data(filename)
    actual_likes = get_actual_likes(df)
    
    kmeans_recommendations = kmeans_recommend(df)
    kmeans_metrics = evaluate_recommendations(kmeans_recommendations, actual_likes)
    print("K-Means Clustering Done.")
    print("K-Means Metrics:", kmeans_metrics)
    
    cf_recommendations = collaborative_filtering(df, False)
    cf_with_sentiment = collaborative_filtering(df, True)
    cf_metrics = evaluate_recommendations(cf_recommendations, actual_likes)
    cf_metrics_sentiment = evaluate_recommendations(cf_with_sentiment, actual_likes)
    print("Collaborative Filtering Done.")
    print("Collaborative Filtering Metrics:", cf_metrics)
    print("Collaborative Filtering with Sentiment Metrics:", cf_metrics_sentiment)
    
    bayes_recommendations = bayesian_recommend(df)
    print("Bayesian Recommendation Done.")
    print("Top Bayesian Recommendations:", bayes_recommendations)