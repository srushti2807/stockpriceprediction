# Step 1: Use an official Python image as the base image
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container
COPY . /app

# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Expose the port that Streamlit will run on
EXPOSE 8501

# Step 6: Set the environment variable for Streamlit to run in headless mode
ENV STREAMLIT_SERVER_HEADLESS true

# Step 7: Run Streamlit when the container starts
CMD ["streamlit", "run", "stock.py"]
