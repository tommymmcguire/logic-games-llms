install:
	pip install -r requirements.txt

lint:
	pylint your_python_module.py

test:
	python -m unittest discover

run:
	python your_script.py

