# Objects-based-imitation-learning-for-autonomous-driving 🚗💡

This repository contains the implementation of an object-based imitation learning approach that utilizes an RNN-based method to predict future waypoints by leveraging object data, lane data, and IMU data. It includes modules for data processing, visualization, and a transformer-based imitation learning model.

---

## 📦 **Installation Guide**
Follow the steps below to set up the environment and install all dependencies.

### **1️⃣ Create a Virtual Environment**
It is recommended to use a virtual environment to manage dependencies.

```bash
python -m venv aha
source aha/bin/activate   # Linux/macOS
```

### **2️⃣ Install Dependencies**
Run the following command to install all dependencies:

```bash
pip install .
```

This will:
- Read `requirements.txt` and install all required packages.
- Install the project as a package for easy imports.

---

## 🚀 **Usage**
### **1️⃣ Running Data Visualization**
You can visualize waypoints, and IMU data, etc. using:

```bash
python imitationLearning/utils/visualization/visualize_data.py 
```

### **2️⃣ Training the Model**
To train the transformer-based imitation learning model:

```bash
python main.py
```
Checkpoints will be saved to the `output/` directory.

### **3️⃣ Testing the Model**
Once training is completed, evaluate the model’s performance using:
```bash
python main_test_model.py
```

### **4️⃣ Running Ouput Visualization**
To visualize the model’s predictions:

```bash
python imitationLearning/utils/visualization/visualize_output.py
```

---

⭐ For more detailed examples and sample outputs, refer to the files in ./docs/samples/

## 📁 **Project Structure**


```
Objects-based-imitation-learning-for-autonomous-driving/
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
│   └── trajectory_visualization.png
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
