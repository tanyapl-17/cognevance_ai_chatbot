# 🤖 Nexa AI — Smart Chatbot

An interactive AI-powered chatbot built with Python, Streamlit, and Natural Language Processing (NLP). Nexa AI identifies user intent from messages and responds using predefined conversational responses.

## ✨ Features

- 💬 Interactive chat interface
- 🧠 NLP-based intent classification
- 📊 Prediction confidence display
- ⚡ Quick-prompt suggestions
- 🗂️ Multiple conversational topics
- 🔄 Clear conversation option

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- JSON

## 📁 Project Structure

```text
cognevance_ai_chatbot/
├── app.py
├── train_model.py
├── intents.json
├── requirements.txt
├── .gitignore
├── README.md
└── chatbot_model.pkl
```

## 🚀 How to Run

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Train the chatbot model

```bash
python train_model.py
```

### 3. Launch the application

```bash
python -m streamlit run app.py
```

## 🧠 How It Works

1. User enters a message.
2. TF-IDF converts the text into numerical features.
3. Logistic Regression predicts the likely intent.
4. The chatbot selects a response associated with that intent.
5. If confidence is too low, it provides a fallback response.

## 🎯 Project Objective

To demonstrate basic NLP, text classification, and deployment of an interactive chatbot through a web-based interface.

## 👩‍💻 Author

Developed as part of an AI/ML internship project.

## 📌 Future Improvements

- Add more training examples and conversational intents.
- Improve model evaluation and response accuracy.
- Add persistent chat history.
- Explore integration with a language model API.