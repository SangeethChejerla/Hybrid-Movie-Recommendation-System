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


Here are the modified paths for downloading, unzipping, and moving the CSV files as per your request:

bash
Copy code
# Download the dataset using the Kaggle CLI and save it to the 'archive' directory
RUN kaggle datasets download -d rounakbanik/the-movies-dataset -p app/archive

# Unzip the downloaded dataset and move the required CSV files to the 'archive' directory
RUN unzip app/archive/the-movies-dataset.zip -d app/archive \
    && mv app/archive/credits.csv app/archive/ \
    && mv app/archive/keywords.csv app/archive/ \
    && mv app/archive/links_small.csv app/archive/ \
    && mv app/archive/movies_metadata.csv app/archive/ \
    && mv app/archive/ratings_small.csv app/archive/ \
    && rm -rf app/archive/dataset

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


