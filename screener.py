import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# STEP 1: Job description
# ---------------------------
job_description = """
We are looking for a Python Developer with knowledge of data structures, 
OOP concepts, APIs, and database handling. Experience in Flask/Django 
will be an added advantage.
"""

# ---------------------------
# STEP 2: Read resumes (from 'resumes' folder)
# ---------------------------
resumes_folder = "resumes"   # put all your .txt resumes inside this folder
resume_texts = []
resume_names = []

for file in os.listdir(resumes_folder):
    if file.endswith(".txt"):
        with open(os.path.join(resumes_folder, file), "r", encoding="utf-8") as f:
            resume_texts.append(f.read())
            resume_names.append(file)

# ---------------------------
# STEP 3: TF-IDF Vectorization
# ---------------------------
documents = [job_description] + resume_texts
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(documents)

# ---------------------------
# STEP 4: Cosine Similarity
# ---------------------------
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
similarity_scores = cosine_sim.flatten()

# ---------------------------
# STEP 5: Rank resumes
# ---------------------------
ranked_resumes = sorted(
    list(zip(resume_names, similarity_scores)),
    key=lambda x: x[1],
    reverse=True
)

# ---------------------------
# STEP 6: Display Results
# ---------------------------
print("🔹 Ranking of resumes based on job description:\n")
for i, (name, score) in enumerate(ranked_resumes, start=1):
    print(f"{i}. {name} → Match Score: {round(score*100, 2)}%")
