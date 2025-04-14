pipeline {
    agent any

    environment {
        PYTHON_PATH = 'C:\\Users\\srushti jadhav\\AppData\\Local\\Programs\\Python\\Python310\\python.exe'
    }

    stages {
        stage('Clone Repository') {
            steps {
                git branch: 'main', url: 'https://github.com/srushti2807/stockpriceprediction.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                script {
                    bat "\"${env.PYTHON_PATH}\" -m pip install --upgrade pip"
                    bat "\"${env.PYTHON_PATH}\" -m pip install -r requirements.txt"
                }
            }
        }

        stage('Run Script') {
            steps {
                script {
                    // Run the main Python script
                    bat "\"${env.PYTHON_PATH}\" stock.py"
                }
            }
        }

        stage('Run Tests') {
            steps {
                echo 'Running tests...'
                // Optional: Add actual test command here
                // bat "\"${env.PYTHON_PATH}\" -m unittest discover tests"
            }
        }

        stage('Run Streamlit App') {
            steps {
                echo 'Launching Streamlit app...'
                bat "\"${env.PYTHON_PATH}\" -m streamlit run stock.py"
            }
        }

        stage('Post Actions') {
            steps {
                echo 'Pipeline execution completed.'
            }
        }
    }
}
