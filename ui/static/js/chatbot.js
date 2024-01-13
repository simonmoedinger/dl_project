document.addEventListener('DOMContentLoaded', function() {
    var sendButton = document.getElementById('send-button');
    var model_dropdown = document.getElementById('models');
    var chatInput = document.getElementById('user-input');
    var messagesContainer = document.getElementById('chat-container');

    sendButton.addEventListener('click', function() {
        var userMessage = chatInput.value.trim();

        if(userMessage !== "") {
            // Display the user's message
            addMessageToDisplay("You", userMessage);

            // Clear the input field
            chatInput.value = '';

            // Send the user's message to the server
            getChatbotResponse(userMessage);
        }
    });

    model_dropdown.addEventListener('change', function() {
        value = model_dropdown.value
        addMessageToDisplay("Bot", "Lädt Modell " + value + " ... ");
        fetch('/model_select', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 'model': value }),
        })
        .then(response => {
            addMessageToDisplay("Bot", "Modell " + value + " geladen!");
            return response.json()
        })
        .catch((error) => {
            console.error('Error:', error);
            // Display an error message if the bot can't process the message
            addMessageToDisplay("Bot", "Model selection failed.");
        });

    });

    // Listen for 'Enter' keypress on input to trigger sending message
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendButton.click(); // Trigger the send button click event
        }
    });

    function addMessageToDisplay(sender, message) {
        var messagesContainer = document.getElementById('messages');
      
        // Create a new message element
        var messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        if (sender === "You") {
          messageDiv.classList.add('user');
        } else {
          messageDiv.classList.add('bot');
        }
        messageDiv.textContent = message;
        
        // Append the new element to the message container
        messagesContainer.appendChild(messageDiv);
      
        // Scroll to the bottom of the chat box
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
      }

    function getChatbotResponse(message) {
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 'message': message }),
        })
        .then(response => response.json())
        .then(data => {
            // Display the bot's response
            addMessageToDisplay("Bot", data.message);
        })
        .catch((error) => {
            console.error('Error:', error);
            // Display an error message if the bot can't process the message
            addMessageToDisplay("Bot", "Sorry, I can't process your message at the moment.");
        });
    }
    document.getElementById("button1").addEventListener('click', function () {
        var userInput = document.getElementById('user-input');
        document.getElementById('user-input').value = '';
        if (document.getElementById('user-input').value == document.getElementById("button1").value){

        }
        else
            userInput.value +=  ' ';
            userInput.value +=  document.getElementById("button1").value;
    });
    document.getElementById("button2").addEventListener('click', function () {
        var userInput = document.getElementById('user-input');
        document.getElementById('user-input').value = '';
        if (document.getElementById('user-input').value == document.getElementById("button2").value){

        }
        else
            userInput.value +=  ' ';
            userInput.value +=  document.getElementById("button2").value;
    });
    document.getElementById("button3").addEventListener('click', function () {
        var userInput = document.getElementById('user-input');
        document.getElementById('user-input').value = '';
        if (document.getElementById('user-input').value == document.getElementById("button3").value){

        }
        else
            userInput.value +=  document.getElementById("button3").value;
    });
    document.getElementById("button4").addEventListener('click', function () {
        var userInput = document.getElementById('user-input');
        document.getElementById('user-input').value = '';
        if (document.getElementById('user-input').value == document.getElementById("button4").value){

        }
        else
            userInput.value +=  ' ';
            userInput.value +=  document.getElementById("button4").value;
    });
});