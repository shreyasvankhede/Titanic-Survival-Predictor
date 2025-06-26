# 🚢 Titanic Survival Predictor (Gradio App)

This project uses a machine learning model (Logistic Regression) to predict whether a person would have survived the Titanic disaster based on key attributes like passenger class, gender, and age. The user interacts with the prediction model using a clean and simple Gradio interface, and receives a prediction along with a relevant GIF representing the result.

---

## 🔍 Features

- Predict survival status based on:
  - Passenger class (1st, 2nd, 3rd)
  - Gender (Male/Female)
  - Age
- Gradio-based UI for interactive input
- Visual feedback using GIFs for predicted outcomes

---

## 💡 How It Works

- The model is trained on the **Titanic dataset** (`Data/Titanic-Dataset.csv`)
- Data preprocessing includes:
  - Dropping irrelevant columns
  - Label encoding for gender
  - Imputing missing age values
- A **Logistic Regression** model is trained using `scikit-learn`
- User inputs are passed to the model through Gradio sliders and radio buttons

---

## 📸 Sample Output

![Prediction Result GIF](Data/Images/1.gif)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/titanic-survival-predictor.git
cd titanic-survival-predictor

2. Install Dependencies

pip install -r requirements.txt

3. Run the App

python titanic_model.py
```


📁 File Structure
```
titanic-survival-predictor/
│
├── Data/
│   ├── Titanic-Dataset.csv
│   └── Images/
│       ├── 0.gif
│       └── 1.gif
│
├── titanic_model.py
├── README.md
└── requirements.txt
```

📜 License

This project is open-source and free to use.


---

### 📦 `requirements.txt`

```txt
pandas
numpy
gradio
scikit-learn