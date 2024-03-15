# Use the official Python image as a base image
FROM python:3.11

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download MovieLens dataset
RUN mkdir data \
    && wget -O data/ml-latest.zip http://files.grouplens.org/datasets/movielens/ml-latest.zip \
    && unzip data/ml-latest.zip -d data/ \
    && rm data/ml-latest.zip

# Copy the templates directory into the container at /app/templates
COPY templates /app/templates

# Define environment variable
ENV FLASK_APP=app.py

# Expose port 5000
EXPOSE 5000

# Command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]
