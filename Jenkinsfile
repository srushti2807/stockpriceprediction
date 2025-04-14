pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
    }

    stages {
        stage('Clone Repository') {
            steps {
                git 'https://github.com/srushti2807/stockpriceprediction.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                    python3 -m venv $VENV_DIR
                    source $VENV_DIR/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Run Tests') {
            steps {
                sh '''
                    source $VENV_DIR/bin/activate
                    # If you have any test files, run them here
                    echo "No tests yet, skipping..."
                '''
            }
        }

        stage('Run Streamlit App') {
            steps {
                sh '''
                    source $VENV_DIR/bin/activate
                    nohup streamlit run stock.py &
                '''
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
