#!/bin/bash

uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &

streamlit run app/frontend/streamlit_app.py \
--server.port $PORT \
--server.address 0.0.0.0