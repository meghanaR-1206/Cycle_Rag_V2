from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from openai import OpenAI

app = Flask(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize global variables
vectordb = None
client = None
chat_history = []

def initialize_rag_system():
    """Initialize the RAG system"""
    import gdown

    global vectordb, client

    # Set up Groq-compatible OpenAI client
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY environment variable not set")
        return False

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )

    # Load existing chroma_db_books database or download from Drive
    try:
        embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

        db_path = "./chroma_db_books"  # Local path after download

        if not os.path.exists(db_path) or not os.path.isdir(db_path):
            print("📦 chroma_db_books not found, downloading from Google Drive...")
            drive_url = "https://drive.google.com/drive/folders/11Mth9UOl6C1f5Qy592OH9NwXRmr_dscx"
            gdown.download_folder(drive_url, quiet=False, use_cookies=False)

        if os.path.exists(db_path) and os.path.isdir(db_path):
            vectordb = Chroma(persist_directory=db_path, embedding_function=embedding)
            print("✅ Loaded chroma_db_books vector database")
            return True
        else:
            print("❌ Failed to load chroma_db_books after attempted download")
            return False

    except Exception as e:
        print(f"❌ Error initializing RAG system: {str(e)}")
        return False


def get_rag_response(query):
    """Get response from RAG system"""
    global vectordb, client, chat_history
    
    # Predefined casual responses with Maya's personality
    casual_responses = {
        "hi": "Hey there! 🌸 I'm Maya, your cycle companion! I'm here to chat about all things periods and menstrual health. What's on your mind today?",
        "hello": "Hello lovely! 💕 Ready to dive into some period talk? I'm here to help with any questions about your cycle or women's health!",
        "hey": "Hey! 😊 So excited to chat with you! Whether it's cramps, cycles, or just curious questions - I'm your girl. What can I help you with?",
        "how are you": "I'm doing amazing, thanks for asking! 🌟 I'm always energized when I get to help someone learn about their body. How are YOU feeling about your cycle lately?",
        "what can you do": "Oh, I'm so glad you asked! 🤗 I'm like your knowledgeable bestie who loves talking about periods, cycles, PMS, cramps, hormones - basically anything related to your amazing reproductive health! Think of me as your personal period guru with a side of encouragement! ✨",
        "who are you": "I'm Maya! 🌸 Think of me as that friend who's been through ALL the period struggles and came out the other side with tons of knowledge and a good sense of humor. I'm here to make your menstrual health journey feel less mysterious and more empowering! Your cycle companion, at your service! 💪",
    }
    
    # Check for casual input
    normalized_query = query.strip().lower()
    if normalized_query in casual_responses:
        return {
            "response": casual_responses[normalized_query],
            "context": "",
            "is_casual": True
        }
    
    # Retrieve relevant documents
    docs_and_scores = vectordb.similarity_search_with_score(query, k=4)
    context_chunks = [doc.page_content for doc, score in docs_and_scores if score < 0.8]
    
    if not context_chunks:
        context_chunks = [doc.page_content for doc, score in docs_and_scores]  # fallback
    
    context = "\n".join(context_chunks[:3])  # limit context size
    
    # Add user query to chat history
    chat_history.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}"
    })
    
    # Construct messages for the LLM
    messages = [
        {
            "role": "system",
            "content": (
                "You are Maya, a warm and knowledgeable menstrual health companion! 🌸 Think of yourself as that supportive friend who's been through it all and loves to share wisdom with a smile.\n\n"
                
                "YOUR PERSONALITY:\n"
                "• Warm, empathetic, and genuinely caring - like talking to your best friend\n"
                "• Educational but never preachy - you make learning feel natural and fun\n"
                "• Sprinkle in gentle humor when appropriate (periods can be tough, a little laughter helps!)\n"
                "• Use encouraging phrases like 'You've got this!', 'That's totally normal', 'Your body is amazing'\n"
                "• Include relevant emojis naturally - 🌸💕✨🤗 (but don't overdo it!)\n\n"
                
                "STAY FOCUSED: Only discuss menstrual cycles, periods, women's reproductive health, and related wellness topics. If asked about anything else, gently redirect with humor: 'I'm all about cycles and self-care! Let's talk periods instead 😊'\n\n"
                
                "HOW TO RESPOND:\n"
                "• Start with validation: 'That's such a great question!' or 'I totally get why you're wondering about this'\n"
                "• Give clear, factual info in simple terms (like explaining to a friend over coffee)\n"
                "• Share practical tips when relevant\n"
                "• End with encouragement or a supportive note\n"
                "• If unsure, say something like: 'Hmm, that's a great question! I'd love for you to check with a healthcare provider who can give you personalized advice 💕'\n\n"
                
                "TONE EXAMPLES:\n"
                "Instead of: 'Menstrual cycles typically last 28 days'\n"
                "Say: 'Most cycles hang out around 28 days, but honestly? Anywhere from 21-35 days is totally normal! Your body has its own rhythm 🌸'\n\n"
                
                "Remember: You're not just sharing facts - you're being a supportive companion on their health journey! Make them feel heard, validated, and empowered. 💪✨"
            )
        }
    ] + chat_history[-10:] + [  # Keep last 10 exchanges
        {
            "role": "user",
            "content": f"CONTEXT:\n{context.strip()}\n\nQUESTION:\n{query.strip()}"
        }
    ]
    
    # Get response from LLM
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages
        )
        
        answer = response.choices[0].message.content
        
        # Add assistant's response to chat history
        chat_history.append({"role": "assistant", "content": answer})
        
        # Limit chat history
        chat_history = chat_history[-10:]
        
        return {
            "response": answer,
            "context": context[:200] + "..." if len(context) > 200 else context,
            "is_casual": False
        }
        
    except Exception as e:
        return {
            "response": "I'm sorry, I'm having trouble processing your request right now. Please try again.",
            "context": "",
            "error": str(e)
        }

@app.route('/')
def index():
    """Serve the main chat interface"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cycle Companion Chat</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 15px 20px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            text-align: center;
            box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
        }

        .header h1 {
            color: #4a5568;
            font-size: 1.4rem;
            font-weight: 600;
        }

        .header p {
            color: #718096;
            font-size: 0.9rem;
            margin-top: 5px;
        }

        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 100%;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }

        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .message {
            max-width: 85%;
            padding: 12px 16px;
            border-radius: 18px;
            font-size: 15px;
            line-height: 1.4;
            animation: slideIn 0.3s ease-out;
        }

        .user-message {
            align-self: flex-end;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-bottom-right-radius: 8px;
        }

        .bot-message {
            align-self: flex-start;
            background: rgba(255, 255, 255, 0.9);
            color: #2d3748;
            border-bottom-left-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        .context-preview {
            font-size: 12px;
            color: #718096;
            background: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            border-left: 3px solid #667eea;
            max-width: 85%;
            align-self: flex-start;
        }

        .input-container {
            padding: 20px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-top: 1px solid rgba(255, 255, 255, 0.2);
        }

        .input-row {
            display: flex;
            gap: 10px;
            align-items: flex-end;
        }

        .input-field {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid rgba(102, 126, 234, 0.3);
            border-radius: 20px;
            font-size: 16px;
            outline: none;
            background: white;
            transition: all 0.3s ease;
            min-height: 44px;
            resize: none;
            font-family: inherit;
        }

        .input-field:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .send-button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            border: none;
            color: white;
            width: 44px;
            height: 44px;
            border-radius: 50%;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }

        .send-button:hover:not(:disabled) {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }

        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .typing-indicator {
            display: none;
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 18px;
            border-bottom-left-radius: 8px;
            align-self: flex-start;
            max-width: 85%;
        }

        .typing-dots {
            display: inline-flex;
            gap: 4px;
        }

        .typing-dots span {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #667eea;
            animation: typing 1.4s infinite;
        }

        .typing-dots span:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-dots span:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes typing {
            0%, 60%, 100% {
                transform: translateY(0);
            }
            30% {
                transform: translateY(-10px);
            }
        }

        .error-message {
            background: #fed7e2;
            color: #c53030;
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: center;
            max-width: 85%;
            align-self: flex-start;
        }

        /* Mobile optimizations */
        @media (max-width: 768px) {
            .messages {
                padding: 15px;
            }
            
            .message {
                max-width: 90%;
                font-size: 14px;
            }
            
            .input-container {
                padding: 15px;
            }
            
            .header {
                padding: 12px 15px;
            }
            
            .header h1 {
                font-size: 1.2rem;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌸 Cycle Companion</h1>
        <p>Your friendly menstrual health assistant</p>
    </div>

    <div class="chat-container">
        <div class="messages" id="messages">
            <div class="bot-message">
                Hi there! I'm here to help you understand your cycle. Ask me anything about menstrual health! 💬
            </div>
        </div>

        <div class="typing-indicator" id="typingIndicator">
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>

        <div class="input-container">
            <div class="input-row">
                <textarea 
                    class="input-field" 
                    id="messageInput" 
                    placeholder="Ask me anything about your cycle..."
                    rows="1"
                ></textarea>
                <button class="send-button" id="sendButton" onclick="sendMessage()">
                    ➤
                </button>
            </div>
        </div>
    </div>

    <script>
        const messagesContainer = document.getElementById('messages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const typingIndicator = document.getElementById('typingIndicator');

        // Auto-resize textarea
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });

        // Send message on Enter (but allow Shift+Enter for new lines)
        messageInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        function addMessage(content, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = isUser ? 'message user-message' : 'message bot-message';
            messageDiv.innerHTML = content;
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function addContext(context) {
            if (context && context.trim()) {
                const contextDiv = document.createElement('div');
                contextDiv.className = 'context-preview';
                contextDiv.innerHTML = `📄 Context: ${context}`;
                messagesContainer.appendChild(contextDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
        }

        function showTyping() {
            typingIndicator.style.display = 'block';
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function hideTyping() {
            typingIndicator.style.display = 'none';
        }

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;

            // Add user message
            addMessage(message, true);
            messageInput.value = '';
            messageInput.style.height = 'auto';
            
            // Disable send button and show typing
            sendButton.disabled = true;
            showTyping();

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message })
                });

                const data = await response.json();
                
                hideTyping();
                
                if (data.error) {
                    addMessage(`Error: ${data.error}`, false);
                } else {
                    // Show context if it's not a casual response
                    if (!data.is_casual && data.context) {
                        addContext(data.context);
                    }
                    addMessage(data.response, false);
                }

            } catch (error) {
                hideTyping();
                addMessage("Sorry, I'm having trouble connecting. Please try again.", false);
                console.error('Error:', error);
            } finally {
                sendButton.disabled = false;
            }
        }

        // Focus input on load
        window.addEventListener('load', () => {
            messageInput.focus();
        });
    </script>
</body>
</html>
    '''

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    data = request.get_json()
    message = data.get('message', '')
    
    if not message:
        return jsonify({"error": "No message provided"}), 400
    
    if not vectordb:
        return jsonify({"error": "RAG system not initialized"}), 500
    
    result = get_rag_response(message)
    return jsonify(result)

if __name__ == '__main__':
    print("🚀 Initializing RAG system...")
    if initialize_rag_system():
        print("✅ RAG system initialized successfully!")
        print("🌐 Starting Flask server...")
        print("📱 Open http://localhost:5000 in your browser")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Failed to initialize RAG system. Please check your book.txt file and environment variables.")