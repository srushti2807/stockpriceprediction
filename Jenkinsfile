pipeline {
    agent any

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
                    // Ensure a clean virtual environment
                    bat '''
                        IF EXIST "%VENV_DIR%" rmdir /S /Q %VENV_DIR%
                        python -m venv %VENV_DIR%
                        call %VENV_DIR%\\Scripts\\activate.bat
                        pip install --upgrade pip
                        pip install -r requirements.txt
                    '''
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    // Add any test running steps here
                    bat '''
                        call %VENV_DIR%\\Scripts\\activate.bat
                        REM Example test run command (adjust if you have test scripts)
                        echo "No tests yet, skipping..."
                    '''
                }
            }
        }

        stage('Run Streamlit App') {
            steps {
                script {
                    // Run Streamlit app in the background
                    bat '''
                        call %VENV_DIR%\\Scripts\\activate.bat
                        start /B streamlit run stock.py > streamlit.log 2>&1
                    '''
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
