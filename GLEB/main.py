from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd


users_file_path = 'C:/Users/ekart/GLEB/14_Mar_2024_17_03_users_report.xlsx'
users_df = pd.read_excel(users_file_path)


filtered_users_df = users_df.dropna(subset=["Персональные качества и навыки", "Знание специализированных программ"], how='all')


original_count = users_df.shape[0]
filtered_count = filtered_users_df.shape[0]

filtered_users_df['Interests_and_Skills'] = filtered_users_df[['Персональные качества и навыки', 'Знание специализированных программ']].apply(lambda x: ' '.join(x.dropna()), axis=1)


projects_file_path = 'C:/Users/ekart/GLEB/14_Mar_2024_20_10_project_report.xlsx'

all_sheets_df = pd.concat(pd.read_excel(projects_file_path, sheet_name=None), ignore_index=True)



clean_projects_df = all_sheets_df.dropna(how='all')

description_column = clean_projects_df.columns[clean_projects_df.applymap(lambda x: isinstance(x, str) and len(x) > 50).any()]

description_column_name = description_column[0] if len(description_column) > 0 else "Column not found"
sample_descriptions = clean_projects_df[description_column].head() if description_column_name != "Column not found" else "No descriptions found"


clean_projects_df = all_sheets_df.dropna(how='all')

description_column = clean_projects_df.columns[clean_projects_df.applymap(lambda x: isinstance(x, str) and len(x) > 50).any()]

description_column_name = description_column[0] if len(description_column) > 0 else "Column not found"
sample_descriptions = clean_projects_df[description_column].head() if description_column_name != "Column not found" else "No descriptions found"



user_interests_skills = filtered_users_df['Interests_and_Skills'].tolist()
project_descriptions = clean_projects_df['Краткое описание проекта'].dropna().tolist()

combined_texts = user_interests_skills + project_descriptions

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_texts)

# Split the TF-IDF matrix back into users and projects
users_tfidf = tfidf_matrix[:len(user_interests_skills)]
projects_tfidf = tfidf_matrix[len(user_interests_skills):]

# Calculate cosine similarity between each user and each project
cosine_similarities = cosine_similarity(users_tfidf, projects_tfidf)

# For each project, find the top 5 matching users
top_matches = {}
for project_idx, project in enumerate(clean_projects_df['Краткое описание проекта'].dropna().index):
    project_name = clean_projects_df.iloc[project]['Название']
    top_user_indices = cosine_similarities[:, project_idx].argsort()[-5:][::-1]
    top_users = filtered_users_df.iloc[top_user_indices][['id', 'Имя']].to_dict(orient='records')
    top_matches[project_name] = top_users

top_matches


def find_top_experts_for_project(project_name):
    """
    Функция для поиска топ-5 пользователей (экспертов) для заданного проекта.

    :param project_name: Название проекта.
    :return: Список из топ-5 пользователей для проекта.
    """
    if project_name not in top_matches:
        return "Проект с таким названием не найден. Пожалуйста, проверьте название и попробуйте снова."
    
    top_users_for_project = top_matches[project_name]
    
    top_users_info = []
    for user in top_users_for_project:
        user_info = f"ID: {user['id']}, Имя: {user['Имя']}"
        top_users_info.append(user_info)
    
    return top_users_info


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union

app = FastAPI()


class User(BaseModel):
    id: int
    Имя: str

@app.get("/top-experts/{project_name}", response_model=List[User])
async def get_top_experts_for_project(project_name: str) -> Union[List[User], str]:
    # Проверяем, есть ли проект с таким названием
    if project_name not in top_matches:
        raise HTTPException(status_code=404, detail="Проект с таким названием не найден. Пожалуйста, проверьте название и попробуйте снова.")
    
    # Получаем топ-5 пользователей для проекта
    top_users_for_project = top_matches[project_name]
    
    # Возвращаем информацию о пользователях
    return top_users_for_project

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")