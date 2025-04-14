pipeline {
    agent any

    environment {
        PYTHON_PATH = '"C:\\Users\\srushti jadhav\\AppData\\Local\\Programs\\Python\\Python310\\python.exe"'
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
                    bat "${env.PYTHON_PATH} -m pip install --upgrade pip"
                    bat "${env.PYTHON_PATH} -m pip install -r requirements.txt"
                }
            }
        }

        stage('Run Script') {
            steps {
                script {
                    bat "${env.PYTHON_PATH} stock.py"
                }
            }
        }

        stage('Run Tests') {
            steps {
                echo 'Running pytest tests...'
                bat "${env.PYTHON_PATH} -m pytest --maxfail=1 --disable-warnings -q"
            }
        }

        stage('Run Streamlit App') {
            steps {
                echo 'Launching Streamlit app...'
                bat 'start "" cmd /c "streamlit run stock.py"'
            }
        }
    }

    post {
        always {
            echo 'Pipeline execution completed.'
        }
        success {
            echo '✅ All steps succeeded!'
        }
        failure {
            echo '❌ One or more steps failed. Check logs.'
        }
    }
}
