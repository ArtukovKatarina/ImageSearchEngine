# Isearch - App for searching images by text or by another image

## Overview
Isearch is a web application that allows users to search for images either by providing a text description or by uploading another image.

## Prerequisites
Before starting, ensure you have the following installed:

1. Python (for the backend)
2. Node.js and npm (for the frontend)
3. Poetry (for managing Python dependencies)

## Getting Started
### Backend setup 
Navigate to the project directory:

```bash
$ cd image_search_engine/
```

Install dependencies inside virtual environment with:

```bash
$ poetry install
```

Start the backend server:

```bash
$ uvicorn main:app --reload
```
After you start server, endpoint OpenAPI specs can be found at
http://127.0.0.1:8000/docs.

### Frontend setup 
Navigate to the frontend directory:

```bash
cd isearch-frontend/
```

Install frontend dependencies: 

```bash
npm install
```

Start the frontend server:

```bash
npm start
```

The frontend will be available at http://localhost:3000/
