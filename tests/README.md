# Running tests

Installing and running tests
```bash
pip install -r requirements
pytest
```

Running tests with gpu
```bash
pytest --gpu
```

Run tests without re-downloading models
```bash
pytest --model-cache ./cache
```

All subsequent runs could be made using
```bash
pytest --model-cache ./cache --skip-download
```
