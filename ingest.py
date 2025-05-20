import pandas as pd
import os
from tqdm import tqdm
import re

TESTING = False
data_dir = 'PENS'

# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("divyapatel4/microsoft-pens-personalized-news-headlines")

# print("Path to dataset files:", path)

# Load datasets
def load_data(data_dir="PENS"):
    for i in tqdm(range(1)):
        train = pd.read_csv(os.path.join(data_dir, 'train.tsv'), sep='\t', header=None)
        valid = pd.read_csv(os.path.join(data_dir, 'valid.tsv'), sep='\t', header=None)
        news = pd.read_csv(os.path.join(data_dir, 'news.tsv'), sep='\t', header=None)
        test = pd.read_csv(os.path.join(data_dir, 'personalized_test.tsv'), sep='\t', header=None)
        news.columns = ['news_id', 'category', 'topic', 'headline', 'body', 'title_entities', 'entity_content']
        train.columns = ['user_id', 'click_news_id', 'dwelltime', 'exposure_time', 'pos', 'neg', 'start_time', 'end_time', 'dwelltime_pos']
        valid.columns = train.columns
        test.columns = ['user_id', 'click_news_id', 'posnew', 'rewrite']
        # Drop unnecessary columns
        test = test.drop(columns=['posnew', 'rewrite'])
        news = news.drop(columns=['title_entities', 'entity_content'])
        news['headline'] = news['headline'].fillna('')
        news['body'] = news['body'].fillna('')
        news['full_text'] = news['headline'] + ' ' + news['body']

    if TESTING:
        train = train.head(1000)
        valid = valid.head(1000)
        test = test.head(1000)
    return train, valid, news, test

# Print basic info
def print_info(train, news, test):
    # Count unique users
    unique_users_train = train['user_id'].nunique()
    print(f"\n# Unique Users in Train: {unique_users_train}")

    # Impressions per user
    impressions_per_user_train = train['user_id'].value_counts()
    print("\nImpressions per user (Train):")
    print(impressions_per_user_train.head(), "\n")


    # Display clicked news ID and exposure time for the first user
    first_user_id = train['user_id'].iloc[1]
    first_user_row = train[train['user_id'] == first_user_id]

    # Step 2: Split click_news_id and exposure_time
    clicks = first_user_row['click_news_id'].iloc[0].split()
    times = first_user_row['exposure_time'].iloc[0].split('#TAB#')

    # Step 3: Create a new DataFrame
    df_first_user = pd.DataFrame({
        'click_news_id': clicks,
        'exposure_time': times
    })

    # Step 4: Display
    print(f"\nClicked News ID and Exposure Time for the First User (User ID: {first_user_id}):")
    print(df_first_user)

    # Example: Breakdown of categories
    print("\nNews categories distribution:")
    print(news['category'].value_counts(), "\n", len(news['category'].value_counts()))
    print("\nNews topic distribution:")
    print(news['topic'].value_counts())

def temporal_split(data, split_time="2019-07-04", train_path="train_split.pkl", test_path="test_split.pkl"):
    # If the pickle files exist, load and return them
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Loading cached train/test splits...")
        train_df = pd.read_pickle(train_path)
        test_df = pd.read_pickle(test_path)
        return train_df, test_df

    # Otherwise, generate the splits
    train_rows = []
    test_rows = []

    for i in tqdm(range(len(data))):
        row = data.iloc[i]
        user_id = row['user_id']
        click_ids = row['click_news_id'].split()
        exposure_times = row['exposure_time'].split('#TAB#')

        for cid, t in zip(click_ids, exposure_times):
            try:
                dt = pd.to_datetime(t)
                if dt < split_time:
                    train_rows.append((user_id, cid, dt))
                else:
                    test_rows.append((user_id, cid, dt))
            except Exception:
                continue

    train_df = pd.DataFrame(train_rows, columns=['user_id', 'click_news_id', 'exposure_time'])
    test_df = pd.DataFrame(test_rows, columns=['user_id', 'click_news_id', 'exposure_time'])

    # Save the new splits if not testing
    if not TESTING:
        train_df.to_pickle(train_path)
        test_df.to_pickle(test_path)

    return train_df, test_df



if __name__ == "__main__":
    train, valid, news, test = load_data(data_dir)


    print_info(valid, news, test)


    valid_split_time = pd.to_datetime("2019-07-04")
    train_df, test_df = temporal_split(valid, valid_split_time)

    #print count of splits of news ids for train and test for first user
    if True:
        first_user_id = train_df['user_id'].iloc[0]
        train_news_count = train_df[train_df['user_id'] == first_user_id]['click_news_id'].value_counts()
        test_news_count = test_df[test_df['user_id'] == first_user_id]['click_news_id'].value_counts()
        print(f"\nTrain news count for first user (User ID: {first_user_id}):")
        print(train_news_count)
        print(f"\nTest news count for first user (User ID: {first_user_id}):")
        print(test_news_count)

        print("\nTrain DataFrame after temporal split:")
        print(train_df.head())
        print("\nTest DataFrame after temporal split:")
        print(test_df.head())


