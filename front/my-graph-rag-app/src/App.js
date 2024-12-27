import React, { useState } from 'react';
import axios from 'axios';
import './styles/App.css';

const App = () => {
  const [userInput, setUserInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleInputChange = (e) => {
    setUserInput(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userInput.trim()) return;

    const userMessage = { sender: 'user', text: userInput };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setUserInput('');
    setIsLoading(true);

    try {
      const res = await axios.post('http://localhost:5002/query', { query: userInput }); 
      const botMessage = { sender: 'bot', text: res.data.response };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      console.error('Error fetching data:', error);
      const errorMessage = { sender: 'bot', text: 'An error occurred while processing your request.' };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🏥 Medical Chatbot Assistant</h1>
        <p>Your trusted AI assistant for healthcare queries</p>
      </header>

      <main className="chat-container">
        <div className="chat-box">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`chat-message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}
            >
              <p>{message.text}</p>
            </div>
          ))}
          {isLoading && (
            <div className="chat-message bot-message">
              <p>Typing...</p>
            </div>
          )}
        </div>

        <form onSubmit={handleSubmit} className="chat-form">
          <input
            type="text"
            value={userInput}
            onChange={handleInputChange}
            placeholder="Type your question here..."
            className="chat-input"
          />
          <button type="submit" className="chat-submit" disabled={isLoading}>
            Send
          </button>
        </form>
      </main>

      <footer className="App-footer">
        <p>Medical Assistant AI © 2024 | Empowering Healthcare with AI</p>
      </footer>
    </div>
  );
};

export default App;
