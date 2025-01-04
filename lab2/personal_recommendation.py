import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import SVD, Dataset, Reader

def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv"""
    df = df[df['Book-Rating'] >= 0]

    book_counts = df['ISBN'].value_counts()
    valid_books = book_counts[book_counts > 1].index
    df = df[df['ISBN'].isin(valid_books)]

    user_counts = df['User-ID'].value_counts()
    valid_users = user_counts[user_counts > 1].index
    df = df[df['User-ID'].isin(valid_users)]
    return df
def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv"""
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


def recommend_books(ratings: pd.DataFrame, svd_model: SVD, linreg_model, books: pd.DataFrame) -> pd.DataFrame:
    """Рекомендовать книги пользователю."""
    user_id = ratings[ratings['Book-Rating'] == 0]['User-ID'].value_counts().idxmax()
    zero_ratings_books = ratings[(ratings['User-ID'] == user_id) & (ratings['Book-Rating'] == 0)]

    recommendations = []
    for _, row in zero_ratings_books.iterrows():
        pred = svd_model.predict(user_id, row['ISBN'])
        if pred.est >= 8:
            recommendations.append({'ISBN': row['ISBN'], 'svd_rating': pred.est})

    if not recommendations:
        return pd.DataFrame([])

    recommendations_df = pd.DataFrame(recommendations)

    merged = pd.merge(recommendations_df, books, on='ISBN', how='left')
    merged = merged[merged['Book-Title'].notna() & (merged['Book-Title'] != '')]

    with open("tfidf.pkl", "rb") as tfidf_file:
        tfidf = pickle.load(tfidf_file)

    title_vectors = tfidf.transform(merged['Book-Title']).toarray()

    merged['Book-Author'] = merged['Book-Author'].astype('category').cat.codes
    merged['Publisher'] = merged['Publisher'].astype('category').cat.codes
    merged['Year-Of-Publication'] = pd.to_numeric(merged['Year-Of-Publication'], errors='coerce')

    features = pd.concat([pd.DataFrame(title_vectors, index=merged.index),
                          merged[['Book-Author', 'Publisher', 'Year-Of-Publication']]], axis=1)

    features.columns = features.columns.astype(str)

    linreg_ratings = linreg_model.predict(features)
    merged['linreg_rating'] = linreg_ratings

    return merged[['Book-Title', 'svd_rating', 'linreg_rating']].sort_values(by='linreg_rating', ascending=False)

if __name__ == "__main__":
    ratings = pd.read_csv("Ratings.csv")
    books = pd.read_csv("Books.csv", low_memory=False)
    ratings = ratings_preprocessing(ratings)
    books = books_preprocessing(books)
    with open("svd.pkl", "rb") as svd_file:
        svd_model = pickle.load(svd_file)
    with open("linreg.pkl", "rb") as linreg_file:
        linreg_model = pickle.load(linreg_file)
    recommendations = recommend_books(ratings, svd_model, linreg_model, books)
    if not recommendations.empty:
        recommendations.to_csv("recommendation.txt", sep="\t", index=False, encoding="utf-8")
    else:
        with open("recommendation.txt", "w", encoding="utf-8") as file:
            file.write("No recommendations available.")