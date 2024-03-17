# Use the official Python image as a base image
FROM python:3.11

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install the Kaggle CLI
RUN pip install kaggle

# Copy your Kaggle API credentials file (kaggle.json) into the container
COPY kaggle.json /root/.kaggle/kaggle.json

# Set permissions for the Kaggle API credentials file
RUN chmod 600 /root/.kaggle/kaggle.json

# Download the dataset using the Kaggle CLI
RUN kaggle datasets download -d rounakbanik/the-movies-dataset -p /app/archive

# Unzip the downloaded dataset
RUN unzip /app/archive/the-movies-dataset.zip -d /app/archive/dataset

# Optionally, remove the downloaded ZIP file
RUN rm /app/archive/the-movies-dataset.zip

# Copy the templates directory into the container at /app/templates
COPY templates /app/templates

# Copy the app.py file into the container at /app
COPY app.py /app/

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable for Flask app entry point
ENV FLASK_APP app.py

# Run the Flask application when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]


