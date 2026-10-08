# AIRFUEL

An applied machine learning project for **aircraft fuel consumption prediction**.

## Overview

AIRFUEL is structured as an end-to-end application rather than a standalone notebook. The repository separates the training workflow, backend services, frontend, configuration, and data/database components so the model can be developed and exposed as an application.

## Project Structure

```text
AIRFUEL/
├── backend/        # Application/backend services
├── frontend/       # User interface
├── training/       # Model training workflow
├── database/       # Data/database layer
├── config/         # Application configuration
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Stack

- Python
- Machine Learning
- Model training and inference
- FastAPI/backend services
- Docker
- Docker Compose
- Frontend web application

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

For the containerized setup:

```bash
docker compose up --build
```

Refer to the source files in `training/`, `backend/`, and `frontend/` for the current application workflow and configuration.

## Project Goal

Build a practical prediction system around aircraft fuel consumption while keeping the machine-learning workflow deployable and easy to extend.

## Author

**Prem Sharma**

GitHub: https://github.com/PremNSharma
