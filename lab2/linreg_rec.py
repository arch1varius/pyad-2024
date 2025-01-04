import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
import joblib


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['Book-Rating'] > 0]

    book_counts = df['ISBN'].value_counts()
    valid_books = book_counts[book_counts > 1].index
    df = df[df['ISBN'].isin(valid_books)]

    user_counts = df['User-ID'].value_counts()
    valid_users = user_counts[user_counts > 1].index
    df = df[df['User-ID'].isin(valid_users)]
    return df
def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[209538, "Book-Author"] = "Michael Teitelbaum"
    df.loc[209538, "Year-Of-Publication"] = 2000
    df.loc[209538, "Book-Title"] = "DK Readers: Creating the X-Men, How It All Began"

    df.loc[220731, "Book-Author"] = "Jean-Marie Gustave"
    df.loc[220731, "Year-Of-Publication"] = 2003
    df.loc[220731, "Book-Title"] = "Peuple du ciel, suivi de Les Bergers"

    df.loc[221678, "Book-Author"] = "James Buckley"
    df.loc[221678, "Year-Of-Publication"] = 2000
    df.loc[221678, "Book-Title"] = "DK Readers: Creating the X-Men, How Comic Books Come to Life"

    df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce')

    df.loc[df['Year-Of-Publication'] > 2024, 'Year-Of-Publication'] = 2024

    df['Year-Of-Publication'] = df['Year-Of-Publication'].astype(int).astype(str)
    df = df.drop([118033, 128890, 129037, 187689])
    df = df.drop(columns=['Image-URL-S'])
    df = df.drop(columns=['Image-URL-M'])
    df = df.drop(columns=['Image-URL-L'])
    return df
def prepare_data(ratings_df, books_df):

    data = pd.merge(ratings_df, books_df, on='ISBN', how='inner')

    avg_ratings = data.groupby('ISBN')['Book-Rating'].mean().reset_index()
    avg_ratings.rename(columns={'Book-Rating': 'Average-Rating'}, inplace=True)

    full_data = pd.merge(books_df, avg_ratings, on='ISBN', how='inner')

    full_data.dropna(subset=['Book-Title', 'Book-Author', 'Publisher', 'Year-Of-Publication'], inplace=True)

    return full_data

def transform_data(data):
    tfidf = TfidfVectorizer(max_features=1000)
    title_vectors = tfidf.fit_transform(data['Book-Title']).toarray()

    with open("tfidf.pkl", "wb") as file:
        pickle.dump(tfidf, file)

    data['Book-Author'] = data['Book-Author'].astype('category').cat.codes
    data['Publisher'] = data['Publisher'].astype('category').cat.codes

    data['Year-Of-Publication'] = pd.to_numeric(data['Year-Of-Publication'], errors='coerce')

    X = pd.concat([
        pd.DataFrame(title_vectors, index=data.index),
        data[['Book-Author', 'Publisher', 'Year-Of-Publication']]
    ], axis=1)

    X.columns = X.columns.astype(str)

    y = data['Average-Rating']

    return X, y, tfidf

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SGDRegressor(max_iter=2000, tol=1e-4, alpha=0.01, penalty='l2', learning_rate='adaptive')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae}')
    if mae < 1.5:
        with open("linreg.pkl", "wb") as file:
            pickle.dump(model, file)
        print(f"Model saved successfully with MAE: {mae}")
    else:
        print(f"Model not saved. MAE is too high: {mae}")

def main():
    books_df = pd.read_csv("Books.csv", low_memory=False)
    ratings_df = pd.read_csv("Ratings.csv")

    books_df = books_preprocessing(books_df)
    ratings_df1 = ratings_preprocessing(ratings_df)

    data = prepare_data(ratings_df1, books_df)

    X, y, tfidf = transform_data(data)
    train_model(X, y)
if __name__ == "__main__":
    main()