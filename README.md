App for searching images by text or by another image

Getting Started
Install dependencies inside virtual environment with

$ poetry install


Start image_search_engine backend with

$ cd image_search_engine/
$ uvicorn main:app --reload

OpenAPI Specs
After you start server, endpoint OpenAPI specs can be found at
[http://127.0.0.1:8000/docs].

To start frontend in React just position in isearch-frontend and run command:
$ npm start
Front will start on [http://localhost:3000/]
