# Steps to obtain predictions

**Clone the repository**

```
git clone https://github.com/amruthaa08/ML_fashion_task.git
```
**Navigate into the cloned directory**

```
cd ML_fashion_task
```

**Build Docker image**
```
docker build -t <image_name> .
```

**Run Server**
```
docker run -p <port>:8080 <image_name>
```

**Using Curl to get predictions**
```
curl -X POST localhost:<port>/predict -F "file=@<path_to_file>"
```
You can also use files provided in the _sample_data_ directory to get predictions. For example,

```
curl -X POST localhost:8000/predict -F "file=@sample_data/1628.jpg"
```
