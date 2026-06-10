import pandas as pd
import numpy as np
import ast
import os
from datetime import datetime
from tkinter import *
from tkinter import messagebox
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

visualization_folder = "visualizations"
if not os.path.exists(visualization_folder):
    os.makedirs(visualization_folder)

print("Visualization Folder Ready")

print("Loading Datasets...")
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
tmdb_movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
print("Datasets Loaded Successfully")

tmdb_movies = tmdb_movies.merge(credits, on='title')
print("Datasets Merged Successfully")

tmdb_movies = tmdb_movies[
    [
        'movie_id',
        'title',
        'overview',
        'genres',
        'keywords',
        'cast',
        'crew',
        'popularity',
        'vote_average'
    ]
]

tmdb_movies.dropna(inplace=True)
print("Null Values Removed")

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

print("Data Preprocessing Started")
tmdb_movies['genres'] = tmdb_movies['genres'].apply(convert)
tmdb_movies['keywords'] = tmdb_movies['keywords'].apply(convert)
tmdb_movies['cast'] = tmdb_movies['cast'].apply(convert_cast)
tmdb_movies['crew'] = tmdb_movies['crew'].apply(fetch_director)
tmdb_movies['overview'] = tmdb_movies['overview'].apply(lambda x: x.split())

tmdb_movies['genres'] = tmdb_movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
tmdb_movies['keywords'] = tmdb_movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
tmdb_movies['cast'] = tmdb_movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
tmdb_movies['crew'] = tmdb_movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

tmdb_movies['tags'] = (
    tmdb_movies['overview'] +
    tmdb_movies['genres'] +
    tmdb_movies['keywords'] +
    tmdb_movies['cast'] +
    tmdb_movies['crew']
)

new_df = tmdb_movies[
    [
        'movie_id',
        'title',
        'tags',
        'popularity',
        'vote_average'
    ]
].copy()

new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: " ".join(x))
print("Tags Created Successfully")

tfidf = TfidfVectorizer(stop_words='english')
vectors = tfidf.fit_transform(new_df['tags']).toarray()
print("TF-IDF Vectorization Completed")

similarity = cosine_similarity(vectors)
print("Cosine Similarity Calculated")

def recommend_movie():
    movie_name = movie_entry.get()
    if movie_name == "":
        messagebox.showerror("Error", "Please Enter Movie Name")
        return
    try:
        movie_name = movie_name.lower().strip()
        new_df['title_lower'] = new_df['title'].str.lower().str.strip()
        matched_movies = new_df[new_df['title_lower'].str.contains(movie_name)]
        if matched_movies.empty:
            messagebox.showerror("Error", "Movie Not Found")
            return
        movie_index = matched_movies.index[0]
        distances = similarity[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        result_box.delete(0, END)
        recommended_movies = []
        similarity_scores = []
        for i in movie_list:
            movie_title = new_df.iloc[i[0]].title
            score = round(i[1], 2)
            recommended_movies.append(movie_title)
            similarity_scores.append(score)
            result_box.insert(END, movie_title + " | Similarity Score: " + str(score))

        plt.figure(figsize=(10, 5))
        plt.bar(recommended_movies, similarity_scores)
        plt.xlabel("Recommended Movies")
        plt.ylabel("Similarity Score")
        plt.title("Top Recommended Movies")
        plt.xticks(rotation=15)
        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_path = os.path.join(visualization_folder, f"recommendation_graph_{timestamp}.png")
        plt.savefig(graph_path)
        print(f"Graph Saved: {graph_path}")
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", str(e))

def save_history():
    movie_name = movie_entry.get()
    if movie_name == "":
        messagebox.showerror("Error", "Please Enter Movie Name")
        return
    with open("recommendation_history.txt", "a") as file:
        file.write(movie_name + "\n")
    messagebox.showinfo("Saved", "Recommendation Saved Successfully")

def clear_all():
    movie_entry.delete(0, END)
    result_box.delete(0, END)

def show_top_movies():
    top_movies = new_df.sort_values(by='popularity', ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    plt.bar(top_movies['title'], top_movies['popularity'])
    plt.xticks(rotation=75)
    plt.xlabel("Movies")
    plt.ylabel("Popularity")
    plt.title("Top 10 Popular Movies")
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_path = os.path.join(visualization_folder, f"top_movies_graph_{timestamp}.png")
    plt.savefig(graph_path)
    print(f"Graph Saved: {graph_path}")
    plt.show()

root = Tk()
root.title("Movie Recommendation System")
root.geometry("1000x700")
root.configure(bg="#121212")

title_label = Label(root, text="MOVIE RECOMMENDATION SYSTEM", font=("Arial", 28, "bold"), bg="#121212", fg="white")
title_label.pack(pady=20)

search_label = Label(root, text="Enter Movie Name", font=("Arial", 18), bg="#121212", fg="white")
search_label.pack()

movie_entry = Entry(root, width=40, font=("Arial", 18))
movie_entry.pack(pady=15)

button_frame = Frame(root, bg="#121212")
button_frame.pack(pady=15)

recommend_button = Button(button_frame, text="Recommend Movies", font=("Arial", 14, "bold"), bg="green", fg="white", padx=20, pady=10, command=recommend_movie)
recommend_button.grid(row=0, column=0, padx=10)

save_button = Button(button_frame, text="Save History", font=("Arial", 14, "bold"), bg="blue", fg="white", padx=20, pady=10, command=save_history)
save_button.grid(row=0, column=1, padx=10)

clear_button = Button(button_frame, text="Clear", font=("Arial", 14, "bold"), bg="red", fg="white", padx=20, pady=10, command=clear_all)
clear_button.grid(row=0, column=2, padx=10)

top_movies_button = Button(button_frame, text="Top Movies", font=("Arial", 14, "bold"), bg="purple", fg="white", padx=20, pady=10, command=show_top_movies)
top_movies_button.grid(row=0, column=3, padx=10)

result_label = Label(root, text="Recommended Movies", font=("Arial", 20, "bold"), bg="#121212", fg="white")
result_label.pack(pady=20)

result_box = Listbox(root, width=90, height=15, font=("Arial", 14), bg="white", fg="black")
result_box.pack(pady=10)

footer_label = Label(root, text="Hybrid Recommendation System Using Machine Learning and Python", font=("Arial", 12), bg="#121212", fg="lightgray")
footer_label.pack(side=BOTTOM, pady=10)

print("GUI Started Successfully")
root.mainloop()