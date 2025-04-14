pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
        PYTHON = '"C:\\Users\\srushti jadhav\\AppData\\Local\\Programs\\Python\\Python310\\python.exe"' // Full path to Python with quotes
    }

    stages {
        stage('Clone Repository') {
            steps {
                git branch: 'main', url: 'https://github.com/srushti2807/stockpriceprediction.git'
            }
        }

        stage('Set up Virtual Environment & Install Dependencies') {
            steps {
                script {
                    bat """
                        IF EXIST %VENV_DIR% rmdir /S /Q %VENV_DIR%
                        %PYTHON% -m venv %VENV_DIR%
                        call %VENV_DIR%\\Scripts\\activate.bat && pip install --upgrade pip
                        call %VENV_DIR%\\Scripts\\activate.bat && pip install -r requirements.txt
                    """
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    bat """
                        call %VENV_DIR%\\Scripts\\activate.bat
                        echo "✅ No automated tests yet."
                    """
                }
            }
        }

        stage('Run Streamlit App') {
            steps {
                script {
                    bat """
                        call %VENV_DIR%\\Scripts\\activate.bat
                        start /MIN streamlit run stock.py --server.headless true > streamlit.log 2>&1
                    """
                }
            }
        }
    }

    post {
        always {
            echo '📝 Pipeline execution completed.'
        }
        success {
            echo '✅ Deployment Successful!'
        }
        failure {
            echo '❌ Deployment Failed.'
        }
    }
}
