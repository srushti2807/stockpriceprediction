pipeline {
    agent any

    stages {
        stage('Declarative: Checkout SCM') {
            steps {
                checkout scm
            }
        }

        stage('Clone Repository') {
            steps {
                git 'https://github.com/srushti2807/stockpriceprediction.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                script {
                    // Ensure pip is upgraded and install dependencies
                    bat '"C:\\Users\\srushti jadhav\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" -m pip install --upgrade pip'
                    bat '"C:\\Users\\srushti jadhav\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" -m pip install -r requirements.txt'
                }
            }
        }

        stage('Run Stock.py') {
            steps {
                script {
                    // Run the stock.py script
                    bat '"C:\\Users\\srushti jadhav\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" stock.py'
                }
            }
        }

        stage('Run Tests') {
            steps {
                // Run your tests if any
            }
        }

        stage('Run Streamlit App') {
            steps {
                // Run the Streamlit app
            }
        }

        stage('Declarative: Post Actions') {
            steps {
                echo 'Pipeline execution completed.'
            }
        }
    }
}
