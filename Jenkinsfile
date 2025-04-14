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
                    sh '''#!/bin/bash
                        if [ -d "$VENV_DIR" ]; then
                            rm -rf $VENV_DIR
                        fi
                        python3 -m venv $VENV_DIR
                        source $VENV_DIR/bin/activate
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
                    sh '''#!/bin/bash
                        source $VENV_DIR/bin/activate
                        # Example test run command (adjust if you have test scripts)
                        echo "No tests yet, skipping..."
                    '''
                }
            }
        }

        stage('Run Streamlit App') {
            steps {
                script {
                    // Run Streamlit app in the background
                    sh '''#!/bin/bash
                        source $VENV_DIR/bin/activate
                        nohup streamlit run stock.py > streamlit.log 2>&1 &
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

