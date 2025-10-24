#!/usr/bin/env python3
"""
Environment verification script for TSLib

This script checks if the environment meets all requirements for running TSLib,
including Python version, Java version (for Spark), and all dependencies.
"""

import sys
import subprocess
import importlib
from typing import Dict, List, Tuple, Optional


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible (3.9-3.13)"""
    version = sys.version_info
    if version >= (3, 9) and version < (3, 14):
        return True, f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible"
    else:
        return False, f"✗ Python {version.major}.{version.minor}.{version.micro} is not compatible. Required: 3.9-3.13"


def check_java_version() -> Tuple[bool, str]:
    """Check if Java version is compatible (17+ for Spark)"""
    try:
        result = subprocess.run(['java', '-version'], 
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            return False, "✗ Java not found. PySpark 4.0.1 requires Java 17+"
        
        # Parse Java version from output
        version_line = result.stdout.split('\n')[0]
        if '"' in version_line:
            version_str = version_line.split('"')[1]
            major_version = int(version_str.split('.')[0])
            
            if major_version >= 17:
                return True, f"✓ Java {version_str} is compatible with PySpark 4.0.1"
            else:
                return False, f"✗ Java {version_str} is not compatible. PySpark 4.0.1 requires Java 17+"
        else:
            return False, "✗ Could not parse Java version"
            
    except FileNotFoundError:
        return False, "✗ Java not found. PySpark 4.0.1 requires Java 17+"
    except Exception as e:
        return False, f"✗ Error checking Java version: {e}"


def check_package(package_name: str, min_version: Optional[str] = None) -> Tuple[bool, str]:
    """Check if a Python package is installed and optionally check version"""
    try:
        module = importlib.import_module(package_name)
        if min_version and hasattr(module, '__version__'):
            from packaging import version
            if version.parse(module.__version__) >= version.parse(min_version):
                return True, f"✓ {package_name} {module.__version__} is installed"
            else:
                return False, f"✗ {package_name} {module.__version__} is too old. Required: {min_version}+"
        else:
            return True, f"✓ {package_name} is installed"
    except ImportError:
        return False, f"✗ {package_name} is not installed"


def check_spark_availability() -> Tuple[bool, str]:
    """Check if PySpark is available and can create a Spark session"""
    try:
        import pyspark
        from pyspark.sql import SparkSession
        
        # Try to create a minimal Spark session
        spark = SparkSession.builder \
            .appName("EnvironmentCheck") \
            .master("local[1]") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        
        spark.stop()
        return True, f"✓ PySpark {pyspark.__version__} is working correctly"
        
    except ImportError:
        return False, "✗ PySpark is not installed"
    except Exception as e:
        return False, f"✗ PySpark is installed but not working: {e}"


def main():
    """Main environment check function"""
    print("TSLib Environment Verification")
    print("=" * 50)
    
    checks = []
    
    # Check Python version
    python_ok, python_msg = check_python_version()
    checks.append(("Python Version", python_ok, python_msg))
    
    # Check Java version
    java_ok, java_msg = check_java_version()
    checks.append(("Java Version", java_ok, java_msg))
    
    # Check core dependencies
    core_deps = [
        ("numpy", "1.24.0"),
        ("scipy", "1.10.0"),
        ("pandas", "1.5.0"),
        ("matplotlib", "3.6.0"),
        ("psutil", "5.9.0"),
    ]
    
    for package, min_version in core_deps:
        dep_ok, dep_msg = check_package(package, min_version)
        checks.append((f"{package.title()}", dep_ok, dep_msg))
    
    # Check Spark dependencies
    spark_deps = [
        ("pyspark", "4.0.1"),
        ("pyarrow", "10.0.0"),
    ]
    
    for package, min_version in spark_deps:
        dep_ok, dep_msg = check_package(package, min_version)
        checks.append((f"{package.title()}", dep_ok, dep_msg))
    
    # Check Spark functionality
    spark_ok, spark_msg = check_spark_availability()
    checks.append(("Spark Functionality", spark_ok, spark_msg))
    
    # Print results
    print("\nEnvironment Check Results:")
    print("-" * 50)
    
    all_passed = True
    for name, passed, message in checks:
        status = "PASS" if passed else "FAIL"
        print(f"{name:<20} {status:<6} {message}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All checks passed! Environment is ready for TSLib.")
        return 0
    else:
        print("✗ Some checks failed. Please install missing dependencies.")
        print("\nInstallation commands:")
        print("  make install        # Install basic dependencies")
        print("  make install-spark  # Install with PySpark support")
        print("  make install-java-macos  # Install Java 17+ (macOS)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
