# 📚 Book Genie — Intelligent Book Recommendation System

> A full-stack **machine learning powered book recommendation system** built using **FastAPI**, **Scikit-Learn**, and **React**, delivering personalized book suggestions through collaborative filtering.

---

## 🚀 Project Overview

**Book Genie** is an end-to-end web application that allows users to discover books intelligently using machine learning.

Users can:
- 🔍 Search books by **title** or **dataset index**
- 🤖 Get **ML-based recommendations** using K-Nearest Neighbors
- 🌙 Switch between **Dark & Light mode**
- 📚 Browse the entire book catalog with images
- ⚡ Experience fast API responses via FastAPI
- 🎨 Use a clean, modern, dashboard-style UI

This project demonstrates **real-world ML deployment**, **API design**, and **frontend-backend integration**.

---

## 🧠 Machine Learning Details

- **Algorithm**: K-Nearest Neighbors (Collaborative Filtering)
- **Similarity Metric**: Cosine Distance
- **Data Representation**:
  - User-Item interaction matrix
  - Sparse matrix for efficiency
- **Model Persistence**:
  - Model, pivot table, and sparse matrix saved using `joblib`
- **Cold-Start Handling**:
  - Case-insensitive search
  - Fuzzy matching fallback
## Run Frontend
```bash
cd book-frontend
npm install
npm start
---

## Run Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
---

## 🏗️ System Architecture

## Author
Built with ❤️ by Khushii