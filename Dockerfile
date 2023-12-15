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

ENV PYTHONPATH "${PYTHONPATH}:/code/app"

# exposing port
EXPOSE 8080

WORKDIR /code/api

# command to run api using uvicorn
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]
