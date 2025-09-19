# 🔧 Proactive Fault Detection for Industrie 4.0

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.2-blueviolet.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)

This project uses machine learning to predict equipment failure from sensor data before it happens. By anticipating breakdowns, this predictive maintenance solution helps reduce costly downtime in manufacturing, a key goal of Industrie 4.0.

---

### 1. 🤔 The Problem

Unplanned equipment downtime is a major source of loss for manufacturers. Traditional maintenance is often inefficient:

* **Reactive:** Fixing equipment only *after* it breaks, causing expensive production stoppages.
* **Preventive:** Servicing equipment on a fixed schedule, which can lead to unnecessary maintenance on healthy parts.

This project tackles the problem by analyzing sensor data to find hidden patterns that signal an impending failure, enabling teams to act proactively.

---

### 2. 💡 My Solution

I built a predictive model using the **AI4I 2020 Predictive Maintenance Dataset** from Kaggle, which contains 10,000 instances with sensor readings like temperature, speed, torque, and tool wear.

My approach involved four key steps:

* **Feature Engineering:** I created new, meaningful features from the raw data. For example, `Power` (Torque × Rotational Speed) and `Temp_diff` (Process Temperature - Air Temperature) gave the model better context.
* **Handling Imbalanced Data:** Failures are rare (only 3.4% of the data), which can bias a model. I used **ADASYN (Adaptive Synthetic Sampling)** to create synthetic examples of failures in the training set, balancing the data.
* **Model Selection:** I evaluated several algorithms and found that **XGBoost** delivered the best performance. I then fine-tuned its hyperparameters using `RandomizedSearchCV`.
* **Optimizing for Recall:** The main goal is to catch as many failures as possible. By lowering the classification threshold to **0.2**, I optimized the model to prioritize finding actual failures (**high Recall**), accepting a moderate number of false alarms (**lower Precision**).

---

### 3. 🏭 Key Insights & Applications

This project demonstrates several principles vital for industrial AI:

* **High Recall is a Business Decision:**
    * Our model achieved a **Recall of 0.84** and a **Precision of 0.54**. This means it successfully identifies 84% of actual failures.
    * **In Practice:** The cost of an unexpected shutdown on an assembly line (e.g., $20,000/minute) is far greater than the cost of a technician checking a false alarm ($100). Our model is optimized for this reality, prioritizing uptime.

* **Engineered Features are More Powerful than Raw Data:**
    * Combining raw sensor data into physically meaningful features like `Power` allows the model to learn more effectively.
    * **In Practice:** A chemical plant could create features representing reaction efficiency or thermal stress, which are better failure indicators than just temperature or pressure alone.

* **You Must Address Data Imbalance:**
    * For rare events like equipment failure, standard models will almost always predict the majority outcome (no failure), making them useless.
    * **In Practice:** Techniques like ADASYN are essential not just for maintenance but also for fraud detection, cybersecurity, and quality control, where the target event is uncommon.

---

### 🚀 Getting Started Locally

Follow these steps to run the project on your machine.

**1. Clone the Repository**

```bash
git clone [https://github.com/yogeshbawankar/Predictive-Maintenance-Industrie4.git](https://github.com/yogeshbawankar/Predictive-Maintenance-Industrie4.git)
cd Predictive-Maintenance-Industrie4
```
**2. Set Up a Virtual Environment**

For **Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

For **macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**

All the necessary libraries are listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

**4. Run the Notebook**

You can now launch the Jupyter Notebook to explore the code.


And that's it! You're all set to explore the data, train the model, and see the results for yourself. 😊
