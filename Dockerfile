FROM tiangolo/uvicorn-gunicorn:python3.6-alpine3.8

#Make directories suited to your application and setting the working directory
RUN mkdir -p /home/project/app

WORKDIR /home/project/app

#Copy and install requirements
COPY requirements.txt /home/project/app
RUN pip install --no-cachedir -r requirements.txt

#Copy contents from your local to your docker container
COPY . /home/project/app