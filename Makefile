.PHONY: test doctor demo clean

doctor:
	python -m dexworld.cli doctor

test:
	python -m unittest discover -s tests -t .

demo:
	python examples/demo_train_mock.py

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
