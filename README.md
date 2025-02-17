# Autobrains Home Assignment 🚗💡

This repository contains the implementation for the Autobrains home assignment. It includes data processing, visualization, and a transformer-based imitation learning model.

---

## 📦 **Installation Guide**
Follow the steps below to set up the environment and install all dependencies.

### **1️⃣ Create a Virtual Environment**
It is recommended to use a virtual environment to manage dependencies.

```bash
python -m venv aha
source aha/bin/activate   # Linux/macOS
aha\Scripts\activate      # Windows (use PowerShell)


### **2️⃣ Install Dependencies**
Run the following command to install all dependencies:

```bash
pip install .
```

This will:
- Read `requirements.txt` and install all required packages.
- Install the project as a package for easy imports.

Alternatively, you can install dependencies manually:

```bash
pip install -r requirements.txt
```

---

## 🚀 **Usage**
### **1️⃣ Running Data Visualization**
You can visualize waypoints, and IMU data, etc. using:

```bash
python visualization/visualize_data.py
```

### **2️⃣ Training the Model**
To train the transformer-based imitation learning model:

```bash
python main.py
```

Checkpoints will be saved to the `output/` directory.

### **3️⃣ Running Ouput Visualization**
To visualize the model’s predictions:

```bash
python visualization/visualize_output.py 
```

---

## 📁 **Project Structure**
```
Autobrains_Home_Assignment/
├── configs
│   ├── model.yaml
│   └── visualization.yaml
├── data
│   ├── images
│   ├── objects
│   └── waypoints
├── imitationLearning
│   ├── data_loader
│   │   ├── data_loader.py
│   │   └── __init__.py
│   ├── __init__.py
│   ├── models
│   │   ├── __init__.py
│   │   └── transformer_rnn_model.py
│   ├── trainers
│   │   ├── il_behaviour_cloner.py
│   │   └── __init__.py
│   └── utils
│       ├── __init__.py
│       └── utils.py
├── main.py
├── output
│   ├── best_model.pth
│   ├── loss.png
│   └── traj  ectory_visualization.png
├── README.md
├── requirements.txt
├── setup.py
├── tests
│   ├── test_dataset.py
│   └── test_model.py
└── visualization
    ├── visualize_data.py
    └── visualize_output.py
```

---
