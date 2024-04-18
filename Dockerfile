# Define the base image
FROM python:3.12.2

# Define the working directory
WORKDIR /app

# Copy the requirements file inside the container
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt && \
    rm requirements.txt

# Copy the whole app inside the container
COPY .  /app

# Define the container port
EXPOSE 8080

# Spin up the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
