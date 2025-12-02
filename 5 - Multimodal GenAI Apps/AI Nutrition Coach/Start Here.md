mkdir cal_coach_app
cd cal_coach_app

python3.11 -m venv my_env
source my_env/bin/activate

pip install ibm-watsonx-ai==1.1.20 image==1.5.33 flask requests==2.32.0