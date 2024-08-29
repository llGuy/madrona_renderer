setup (in python environment):

```
cmake -S . -B build
cd build
make -j
cd -
pip install -e .
```

run test:
```
python scripts/test.py
```
