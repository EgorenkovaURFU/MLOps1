FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py", "--server.port=8501", "server.address=0.0.0.0"]
