# TSLib - Time Series Library Makefile
# Comprehensive build and development automation

.PHONY: help install install-spark install-dev test test-coverage test-spark benchmark clean format lint docs examples check-version

# Default target
help:
	@echo "TSLib - Time Series Library"
	@echo "=========================="
	@echo ""
	@echo "Available targets:"
	@echo "  install        - Install basic dependencies and library"
	@echo "  install-spark  - Install with PySpark support"
	@echo "  install-dev    - Install with development dependencies"
	@echo "  test           - Run all tests"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo "  test-spark     - Run Spark-specific tests"
	@echo "  benchmark      - Run performance benchmarks"
	@echo "  format         - Format code with black"
	@echo "  lint           - Lint code with flake8"
	@echo "  docs           - Generate documentation"
	@echo "  examples       - Run example scripts"
	@echo "  clean          - Clean build artifacts"
	@echo "  check-version  - Check Python version compatibility"
	@echo ""

# Check Python version compatibility
check-version:
	@echo "Checking Python version compatibility..."
	@python3 -c "import sys; version = sys.version_info; assert version >= (3, 9), f'Python 3.9+ required, found {version.major}.{version.minor}'; print(f'✓ Python {version.major}.{version.minor}.{version.micro} is compatible')"

# Check Java version compatibility for Spark
check-java:
	@echo "Checking Java version compatibility for Spark..."
	@if command -v java >/dev/null 2>&1; then \
		JAVA_VERSION=$$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1); \
		if [ "$$JAVA_VERSION" -ge 17 ]; then \
			echo "✓ Java $$JAVA_VERSION is compatible with PySpark 4.0.1"; \
		else \
			echo "✗ Java $$JAVA_VERSION is not compatible. PySpark 4.0.1 requires Java 17+"; \
			echo "  Install Java 17+ using: make install-java-macos"; \
			exit 1; \
		fi; \
	else \
		echo "✗ Java not found. PySpark 4.0.1 requires Java 17+"; \
		echo "  Install Java 17+ using: make install-java-macos"; \
		exit 1; \
	fi

# Installation targets
install: check-version
	@echo "Installing TSLib with basic dependencies..."
	pip install -r requirements.txt
	pip install -e .

install-spark: check-version check-java
	@echo "Installing TSLib with PySpark support..."
	pip install -r requirements.txt
	pip install -r requirements-spark.txt
	pip install -e .

install-dev: check-version
	@echo "Installing TSLib with development dependencies..."
	pip install -r requirements.txt
	pip install -r requirements-spark.txt
	pip install -e ".[dev]"

# Testing targets
test: check-version
	@echo "Running all tests..."
	pytest tests/ -v

test-coverage: check-version
	@echo "Running tests with coverage..."
	pytest tests/ --cov=tslib --cov-report=html --cov-report=term-missing -v

test-spark: check-version check-java
	@echo "Running Spark-specific tests..."
	@export PATH="/opt/homebrew/opt/openjdk@17/bin:$$PATH" && export JAVA_HOME="/opt/homebrew/opt/openjdk@17" && pytest tests/test_spark_comparison.py tests/test_performance_benchmark.py -v

benchmark: check-version check-java
	@echo "Running performance benchmarks..."
	@export PATH="/opt/homebrew/opt/openjdk@17/bin:$$PATH" && export JAVA_HOME="/opt/homebrew/opt/openjdk@17" && python -m pytest tests/test_performance_benchmark.py::TestPerformanceBenchmark::test_comprehensive_benchmark -v -s

# Development targets
format: check-version
	@echo "Formatting code with black..."
	black tslib/ tests/ examples/ --line-length 88

lint: check-version
	@echo "Linting code with flake8..."
	flake8 tslib/ tests/ examples/ --max-line-length 88 --ignore E203,W503

# Documentation targets
docs: check-version
	@echo "Generating documentation..."
	@if [ -d "docs" ]; then \
		echo "Documentation generation not yet implemented"; \
	else \
		echo "Creating basic documentation structure..."; \
		mkdir -p docs; \
		echo "# TSLib Documentation" > docs/README.md; \
		echo "Documentation will be generated here." >> docs/README.md; \
	fi

# Example targets
examples: check-version
	@echo "Running example scripts..."
	@echo "Running basic ARIMA example..."
	python examples/basic_arima.py
	@echo ""
	@echo "Running Spark parallel ARIMA example..."
	@if python -c "import pyspark" 2>/dev/null; then \
		python examples/spark_parallel_arima.py; \
	else \
		echo "PySpark not available, skipping Spark example"; \
	fi
	@echo ""
	@echo "Running internal parallelization demo..."
	python examples/parallel_internal_demo.py

# Cleanup targets
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Development workflow targets
dev-setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation"

quick-test: check-version
	@echo "Running quick test suite..."
	pytest tests/test_arima.py -v

full-test: test test-spark benchmark
	@echo "Full test suite completed!"

# Performance analysis
profile: check-version
	@echo "Running performance profiling..."
	python -m cProfile -o profile_output.prof examples/basic_arima.py
	@echo "Profile saved to profile_output.prof"

# Version and compatibility checks
check-deps: check-version
	@echo "Checking dependency versions..."
	pip list | grep -E "(numpy|scipy|pandas|matplotlib|pyspark)"

# Docker support (if needed)
docker-build:
	@echo "Building Docker image..."
	@if [ -f "Dockerfile" ]; then \
		docker build -t tslib .; \
	else \
		echo "Dockerfile not found"; \
	fi

# CI/CD helpers
ci-test: install-dev test-coverage lint
	@echo "CI test pipeline completed!"

# Release helpers
version-check:
	@echo "Current version information:"
	@python -c "import tslib; print(f'TSLib version: {getattr(tslib, \"__version__\", \"unknown\")}')" 2>/dev/null || echo "TSLib not installed"

# Help for specific targets
help-install:
	@echo "Installation Options:"
	@echo "  make install        - Basic installation (core dependencies only)"
	@echo "  make install-spark  - With PySpark for distributed computing"
	@echo "  make install-dev    - With development tools (pytest, black, etc.)"

help-test:
	@echo "Testing Options:"
	@echo "  make test           - Run all tests"
	@echo "  make test-coverage  - Run tests with coverage report"
	@echo "  make test-spark     - Run Spark-specific tests only (requires Java 17+)"
	@echo "  make benchmark      - Run performance benchmarks (requires Java 17+)"
	@echo "  make quick-test     - Run basic tests only"

# Java installation commands
install-java-macos:
	@echo "Installing Java 17+ on macOS..."
	@if command -v brew >/dev/null 2>&1; then \
		brew install openjdk@17; \
		echo "Java 17 installed. Add to your shell profile:"; \
		echo '  export PATH="/opt/homebrew/opt/openjdk@17/bin:$$PATH"'; \
		echo '  export JAVA_HOME="/opt/homebrew/opt/openjdk@17"'; \
	else \
		echo "Homebrew not found. Please install Java 17+ manually:"; \
		echo "  https://adoptium.net/temurin/releases/?version=17"; \
	fi

install-java-linux:
	@echo "Installing Java 17+ on Linux..."
	@echo "Ubuntu/Debian: sudo apt update && sudo apt install openjdk-17-jdk"
	@echo "CentOS/RHEL: sudo yum install java-17-openjdk-devel"
	@echo "Or download from: https://adoptium.net/temurin/releases/?version=17"

install-java-windows:
	@echo "Installing Java 17+ on Windows..."
	@echo "1. Download from: https://adoptium.net/temurin/releases/?version=17"
	@echo "2. Run the installer"
	@echo "3. Set JAVA_HOME environment variable to installation directory"

help-dev:
	@echo "Development Options:"
	@echo "  make format         - Format code with black"
	@echo "  make lint           - Lint code with flake8"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make dev-setup      - Complete development setup"
