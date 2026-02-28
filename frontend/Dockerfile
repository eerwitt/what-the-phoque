# Stage 1: Build the React frontend
FROM node:20-slim AS frontend-build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: Python server
FROM python:3.11-slim
WORKDIR /app

COPY server-requirements.txt ./
RUN pip install --no-cache-dir -r server-requirements.txt

# Copy the built React app
COPY --from=frontend-build /app/dist ./static

# Copy the FastAPI server
COPY app.py ./

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
