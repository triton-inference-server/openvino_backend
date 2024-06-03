# Running tests

Installing and running tests
```bash
pip install -r requirements.txt
pytest
```

Running tests with gpu
```bash
pytest --gpu
```

Run tests while caching downloaded models
```bash
pytest --model-cache ./cache
```

