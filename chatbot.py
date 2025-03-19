import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import json
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
{
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hello", "Hi", "How are you?", "Is anyone there?", "Good morning"],
            "responses": ["Hello!", "Hi there!", "Good to see you!", "How can I help you?"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "Goodbye", "See you later", "Good night"],
            "responses": ["Goodbye!", "See you later!", "Have a great day!"]
        },
        {
            "tag": "age",
            "patterns": ["How old are you?", "What's your age?", "When were you born?"],
            "responses": ["I am a chatbot, I don't have an age."]
        }
    ]
}
import json

intents_data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hello", "Hi", "How are you?", "Is anyone there?", "Good morning"],
            "responses": ["Hello!", "Hi there!", "Good to see you!", "How can I help you?"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "Goodbye", "See you later", "Good night"],
            "responses": ["Goodbye!", "See you later!", "Have a great day!"]
        },
        {
            "tag": "age",
            "patterns": ["How old are you?", "What's your age?", "When were you born?"],
            "responses": ["I am a chatbot, I don't have an age."]
        }
    ]
}

# Write the intents data to intents.json
with open('intents.json', 'w') as f:
    json.dump(intents_data, f)

# Load intents file after creating it
with open('intents.json') as json_data:
    intents = json.load(json_data)

# ... (rest of your code)
lemmatizer = WordNetLemmatizer()

# Load intents file
with open('intents.json') as json_data:
    intents = json.load(json_data)

# Data preprocessing
patterns = []
responses = []
tags = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        word_list = nltk.word_tokenize(pattern)
        patterns.append(pattern)
        tags.append(intent['tag'])
        responses.append(intent['responses'])

# Lemmatizing and creating a bag of words
words = []
for pattern in patterns:
    word_list = nltk.word_tokenize(pattern)
    words.extend(word_list)

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in nltk.corpus.stopwords.words('english')]
words = sorted(list(set(words)))

# Encode labels (tags)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(tags)

# Save words and labels for later use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(label_encoder, open('label_encoder.pkl', 'wb'))
import nltk

nltk.download('punkt_tab')
# Create a bag of words
vectorizer = TfidfVectorizer(vocabulary=words)
X = vectorizer.fit_transform(patterns).toarray()

# Convert tags to categorical labels
y = np.array(labels)

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Build a sequential model
model = Sequential()
model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(set(tags)), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5')
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in nltk.corpus.stopwords.words('english')]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    return [1 if w in sentence_words else 0 for w in words]

def respond_to_input(input_text):
    # Convert the user's input into a feature vector
    input_data = bow(input_text, words)
    input_data = np.array(input_data).reshape(1, -1)

    # Predict the intent using the trained model
    model = keras.models.load_model('chatbot_model.h5')
    prediction = model.predict(input_data)
    intent_index = np.argmax(prediction)
    tag = label_encoder.inverse_transform([intent_index])[0]

    # Get a random response for the predicted intent
    response = random.choice(responses[tags.index(tag)])
    return response

while True:
    # Chat with the user
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    response = respond_to_input(user_input)
    print(f"Chatbot: {response}")