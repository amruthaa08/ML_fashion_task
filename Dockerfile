# base image
FROM python:3.7

# working dirctory inside container 
WORKDIR /code

# copying requirements into working directory
COPY ./requirements.txt /code/requirements.txt

# installing dpendencies
RUN pip install -r /code/requirements.txt 

# copying files 
COPY ./api /code/api
COPY ./models /code/models

# exposing port
EXPOSE 8080

# command to run api using uvicorn
CMD ["uvicorn", "api.main:app", "--host=0.0.0.0", "--port=8080"]
