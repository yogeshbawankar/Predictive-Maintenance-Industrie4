# 🔧 Proactive Fault Detection for Industrie 4.0

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.2-blueviolet.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)

This project focuses on **proactive fault detection** by developing a robust machine learning model to predict imminent equipment failure from sensor data. By anticipating breakdowns before they occur, this solution directly addresses a core challenge in modern manufacturing and the Industrie 4.0 initiative.

---

### 1. 🤔 What is the Problem We Solve?

Unplanned equipment downtime is a primary source of financial loss and production inefficiency in the manufacturing sector. Traditional maintenance strategies are often:
* **Reactive:** Fixing equipment only after it has failed, causing costly production halts.
* **Preventive:** Servicing equipment on a fixed schedule, which can lead to unnecessary maintenance and underutilization of a component's lifespan.

The core problem is the inability to anticipate machine failures from the vast streams of sensor data available. This project tackles this by building a model that can detect the subtle patterns that precede a failure event, allowing for "predictive maintenance."

---

### 2. 💡 Our Approach & Solution

To solve this, we utilized a systematic data-driven approach based on the **AI4I 2020 Predictive Maintenance Dataset** from Kaggle. This rich dataset contains 10,000 instances with sensor readings like temperature, rotational speed, torque, and tool wear, along with machine failure labels.

Our methodology involved several key steps:

* **Advanced Feature Engineering:** We created new, more insightful features from the raw sensor data, such as `Power` (Torque * Rotational Speed) and `Temp_diff` (Process Temperature - Air Temperature), to provide the model with deeper context.
* **Handling Extreme Class Imbalance:** Machine failures are rare events, making the dataset highly imbalanced (only 3.4% of instances are failures). To overcome this, we used the **ADASYN (Adaptive Synthetic Sampling)** technique to over-sample the minority (failure) class in the training set, creating a more balanced dataset for the model to learn from.
* **Model Selection & Tuning:** We trained and evaluated several models, with **XGBoost** emerging as the top performer. We then used `RandomizedSearchCV` to fine-tune its hyperparameters for optimal performance.
* **Threshold Optimization:** Finally, we adjusted the classification threshold to **0.2**. This crucial step optimized the model to prioritize **Recall** over Precision, ensuring it was highly effective at catching actual failures, which is the primary business goal.

---

### 3. 🏭 Key Insights & Industrial Applications

This project provided several important insights that are directly applicable in an industrial setting:

* **The Precision-Recall Trade-off is a Business Decision:** Our final model achieved a **Recall of 0.84** (catching 84% of actual failures) with a **Precision of 0.54**. This means that while some flagged machines might not fail (false positives), the system is highly effective at preventing unexpected breakdowns.
    * **Industrial Example:** For a critical automotive assembly line, the cost of an unexpected shutdown ($20,000/minute) is astronomical compared to the cost of a technician performing a minor check on a false alarm ($100). Our high-recall model is optimized for this exact business reality, prioritizing production continuity over minimizing minor maintenance checks.

* **Raw Sensor Data is Not Enough:** Creating domain-specific features like `Power` and `Temp_diff` significantly improves a model's predictive power. It allows the model to learn from physical relationships rather than just raw numbers.
    * **Industrial Example:** A chemical processing plant can engineer features that represent reaction efficiency or thermal stress on a vessel. These composite metrics are far more indicative of potential issues than just monitoring temperature or pressure in isolation.

* **Imbalance Handling is Non-Negotiable for Rare Events:** Without a strategy like ADASYN, a model trained on an imbalanced dataset will simply learn to always predict the majority class (no failure), making it useless.
    * **Industrial Example:** This same principle is critical in detecting rare but severe events like cybersecurity breaches or identifying defective products in quality control. Standard models will fail unless specific techniques are used to account for the rarity of the target event.

---

### 🚀 Getting Started Locally

Follow these steps to get your own local copy of this project up and running.

**1. Clone the Repository**

```bash
git clone [https://github.com/yogeshbawankar/Predictive-Maintenance-Pro.git](https://github.com/yogeshbawankar/Predictive-Maintenance-Pro.git)
cd Predictive-Maintenance-Pro
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

```bash
jupyter notebook "predictive-maintenance-eda-analysis.ipynb"
```

And that's it! You're all set to explore the data, train the model, and see the results for yourself. 😊