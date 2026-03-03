.PHONY: install test lint format clean help run-smoke run-vanilla run-pgd run-uat-confsmooth run-uat-ambiguity

help:
	@echo "Available commands:"
	@echo "  make install              - Install dependencies"
	@echo "  make test                 - Run tests"
	@echo "  make lint                 - Check code formatting"
	@echo "  make format               - Format code with black and isort"
	@echo "  make clean                - Remove Python cache files"
	@echo "  make run-smoke            - Run smoke test (2 epochs)"
	@echo "  make run-vanilla          - Train vanilla model (200 epochs)"
	@echo "  make run-pgd              - Train PGD-AT model (200 epochs)"
	@echo "  make run-uat-confsmooth   - Train UAT-ConfSmooth model (200 epochs)"
	@echo "  make run-uat-ambiguity    - Train UAT-Ambiguity model (200 epochs)"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	black --check src/ experiments/ tests/
	isort --check src/ experiments/ tests/

format:
	black src/ experiments/ tests/
	isort src/ experiments/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

run-smoke:
	python experiments/train.py --config experiments/configs/smoke_test.yaml

run-vanilla:
	python experiments/train.py --config experiments/configs/vanilla_cifar10.yaml

run-pgd:
	python experiments/train.py --config experiments/configs/pgd_cifar10.yaml

run-uat-confsmooth:
	python experiments/train.py --config experiments/configs/uat_confsmooth_cifar10.yaml

run-uat-ambiguity:
	python experiments/train.py --config experiments/configs/uat_ambiguity_cifar10.yaml

eval-vanilla:
	python experiments/eval.py \
		--checkpoint checkpoints/vanilla_cifar10/vanilla_final.pt \
		--config experiments/configs/vanilla_cifar10.yaml

eval-pgd:
	python experiments/eval.py \
		--checkpoint checkpoints/pgd_cifar10/pgd_final.pt \
		--config experiments/configs/pgd_cifar10.yaml

eval-uat-confsmooth:
	python experiments/eval.py \
		--checkpoint checkpoints/uat_confsmooth_cifar10/confsmooth_final.pt \
		--config experiments/configs/uat_confsmooth_cifar10.yaml

eval-uat-ambiguity:
	python experiments/eval.py \
		--checkpoint checkpoints/uat_ambiguity_cifar10/class_ambiguity_final.pt \
		--config experiments/configs/uat_ambiguity_cifar10.yaml
