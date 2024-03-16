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

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Copy the templates directory into the container at /app/templates
COPY templates /app/templates

# Define environment variable for Flask app entry point
ENV FLASK_APP app.py

# Run the Flask application when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]

