# Assignment2_recomender_sys

This repository contains the code for creating a recommender system as part of Assignment 2 in the Personalization course.

## Setting Up the Environment

### 1. Create a Virtual Environment

To ensure dependencies are installed in an isolated environment, create a virtual environment using Python:

```sh
python -m venv .venv
```

### 2. Activate the Virtual Environment

- **On Windows** (Command Prompt):
  ```sh
  .venv\Scripts\activate
  ```
  
- **On macOS/Linux**:
  ```sh
  source .venv/bin/activate
  ```

### 3. Install Required Libraries

Once the virtual environment is activated, install the required dependencies from `requirements.txt`:

```sh
pip install -r requirements.txt
```

### 4. Verify Installation

To confirm that all dependencies are installed correctly, run:

```sh
pip list
```

You are now ready to run the recommender system!

## Running the Recommender System

To execute the main script, run:

```sh
python app.py
```

Ensure that all data files are available in the expected paths before running the system.

## Deactivating the Virtual Environment

Once you are done, deactivate the virtual environment by running:

```sh
deactivate
```
