#!bin/bash

. .venv/bin/activate

uvicorn app:api --reload --host 0.0.0.0 --port 8000
