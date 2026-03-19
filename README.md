# Customer Churn Prediction System

This project implements an end-to-end machine learning system to predict customer churn for a SaaS business. It simulates a realistic production workflow, covering data generation, preprocessing, model training, evaluation, and deployment.

---

## Overview

The system is designed to identify customers who are likely to churn based on their behavior and account activity.

### Key Features

- Synthetic dataset generation (7,000+ records with imbalance and missing values)
- Data preprocessing and feature engineering pipeline
- Training and comparison of multiple ML models
- Threshold tuning focused on churn-class performance
- FastAPI-based inference service
- Evaluation artifacts and simple operational setup

---

## Problem

Customer churn is a critical problem in SaaS businesses. Retaining customers is more cost-effective than acquiring new ones, so identifying high-risk users early helps improve retention strategies.

This project predicts whether a customer is likely to churn using behavioral and account-related features.

---

## Features Used

### Core Features

- `usage_frequency` — how often the product is used  
- `subscription_type` — type of subscription plan  
- `login_activity` — login consistency  
- `support_tickets` — number of support interactions  
- `payment_history` — payment reliability  

### Additional Features

- `avg_session_minutes` — average session duration  
- `monthly_spend` — customer spending  
- `tenure_months` — customer lifetime  
- `region` — geographic segment  

---

## Workflow

### 1. Data Generation
- Synthetic dataset with realistic noise, imbalance, and missing values

### 2. Preprocessing
- Handle missing values  
- Encode categorical features  
- Scale numerical features  

### 3. Model Training
- Train multiple models  
- Compare performance  

### 4. Evaluation & Tuning
- Use precision, recall, and F1-score  
- Tune decision threshold for better churn detection  

### 5. Deployment
- FastAPI service with:
  - `/predict` — single prediction  
  - `/predict/batch` — batch prediction  

---

## API Usage

### Single Prediction
`POST /predict`

### Batch Prediction
`POST /predict/batch`

**Input:** JSON with required customer features  
**Output:** Churn probability and predicted label  

---

## Project Structure
.
├── data/                # Generated datasets
├── models/              # Trained models
├── src/
│   ├── preprocessing/   # Data cleaning & feature engineering
│   ├── training/        # Model training scripts
│   ├── evaluation/      # Metrics and evaluation logic
│   └── api/             # FastAPI service
├── artifacts/           # Evaluation outputs
└── README.md


---

## Goal

The goal of this project is to demonstrate how to build a practical, production-style ML system from data generation to deployment, while keeping the implementation simple, clean, and efficient.
