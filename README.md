# Computational Intelligence 2024 - Final Project
---
## Symbolic Regression with Genetic Programming

This project implements a **symbolic regression system** based on **genetic programming**. Using syntax trees, the system evolves mathematical expressions that best approximate a given dataset.



### **Key Features**
- **Genetic Programming (GP)**:
  - Evolution of symbolic expressions.
  - Selection, mutation, and crossover operators.
  - Support for complexity penalties and regularization.
- **Advanced Operator Support**:
  - Arithmetic (`+`, `-`, `*`, `/`, `**`).
  - Trigonometric (`sin`, `cos`, `tan`, etc.).
  - Hyperbolic and logarithmic (`sinh`, `log`, etc.).
- **Modular Configurations**:
  - Create custom operator sets.
  - Manage the number of usable variables.

---

### **Project Structure**
The project is organized into the following main components:

- **`src/main.py`**: 
  The main script to run the system.

- **`src/symb_regression/`**:
  - **`core/`**:
    Contains the implementation of the genetic algorithm:
    - `genetic_programming.py`: Core logic of the evolution process.
    - `tree.py`: Structure and manipulation of syntax trees.
  - **`operators/`**:
    Definition and management of symbolic operators:
    - `definitions.py`: Specifications of unary and binary operators.
    - `crossover.py`: Operations for combining trees.
    - `mutation.py`: Mutation logic for trees.
  - **`utils/`**:
    Supporting functions:
    - `data_handler.py`: Input data handling.
    - `logging_config.py`: Logging configuration.
    - `metrics.py`: Performance metrics (e.g., MSE, R²).
    - `random.py`: Random seed management for reproducibility.
  - **`config/`**:
    Main configurations:
    - `settings.py`: Algorithm parameters.

---

### **Installation**
To run the project, ensure you have Python >= 3.8 and **Poetry** installed.

### **Steps**

#### **1. Install dependencies**
Install the necessary dependencies using Poetry:
```bash
poetry install
```

#### **2. Update dependencies**
Ensure all packages are up-to-date:
```bash
poetry update
```

#### **3. Run the project**
Activate the Poetry environment and execute the main script:
```bash
poetry shell
python src/main.py
```
---
## **Contributes**
- Fiata Rosamaria
- Taormina Nicolò