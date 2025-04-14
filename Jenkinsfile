pipeline {
    agent any

    tools {
        python 'Python3' // Name as set in Global Tool Configuration
    }

    environment {
        VENV_DIR = 'venv'
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
                    bat """
                        IF EXIST "%VENV_DIR%" rmdir /S /Q "%VENV_DIR%"
                        python -m venv %VENV_DIR%
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
                        REM Add your test commands here
                        echo "No tests implemented yet."
                    """
                }
            }
        }

        stage('Run Streamlit App') {
            steps {
                script {
                    bat """
                        call %VENV_DIR%\\Scripts\\activate.bat
                        start /B streamlit run stock.py > streamlit.log 2>&1
                    """
                }
            }
        }
    }

    post {
        success {
            echo '✅ Deployment Successful!'
        }
        failure {
            echo '❌ Deployment Failed.'
        }
    }
}
