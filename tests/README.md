# Running tests

Installing and running tests
```bash
pip install -r requirements.txt
pytest -sv --image=tritonserver:latest
```

Running tests with gpu
```bash
pytest -sv --gpu --image=tritonserver:latest
```

Run tests while caching downloaded models
```bash
mkdir cache
pytest -sv --model-cache ./cache --image=tritonserver:latest
```

