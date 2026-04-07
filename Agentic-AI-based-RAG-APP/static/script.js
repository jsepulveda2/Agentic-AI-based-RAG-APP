document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById("chat-container");
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");
    const pdfBox = document.getElementById("pdf-box");
    const pdfViewer = document.getElementById("pdfViewer");
    const closePdfBtn = document.getElementById("closePdfBtn");

    let activeTyping = null;

    function appendMessage(text, sender, isTyping = false) {
        const msg = document.createElement("div");
        msg.classList.add("message", sender === "user" ? "user-message" : "bot-message");

        const avatar = document.createElement("img");
        avatar.classList.add("avatar");
        avatar.src = sender === "user"
            ? "/static/assets/user.png"
            : "/static/assets/bot.png";

        const box = document.createElement("div");
        box.classList.add("message-text");

        if (isTyping) {
            box.classList.add("typing-indicator");
        } else {
            box.innerHTML = marked.parse(text);
            attachPdfLinks(box);
        }

        msg.appendChild(avatar);
        msg.appendChild(box);
        chatContainer.appendChild(msg);
        chatContainer.scrollTop = chatContainer.scrollHeight;

        return box;
    }

    function attachPdfLinks(element) {
        element.querySelectorAll("a").forEach(link => {
            link.addEventListener("click", e => {
                e.preventDefault();
                openPdf(link.href);
            });
        });
    }

    function openPdf(url) {
        pdfViewer.src = url;
        pdfBox.classList.remove("hidden");
        pdfBox.classList.add("expanded");
    }

    closePdfBtn.addEventListener("click", () => {
        pdfBox.classList.remove("expanded");
        setTimeout(() => {
            pdfBox.classList.add("hidden");
            pdfViewer.src = "";
        }, 300);
    });

    function typewriter(el, text, speed = 20, done = () => {}) {
        let i = 0;
        let plain = document.createElement("div");
        plain.innerHTML = marked.parse(text);
        const finalHTML = plain.innerHTML;
        const raw = plain.innerText;

        function step() {
            if (i < raw.length) {
                el.innerText = raw.substring(0, i + 1);
                i++;
                chatContainer.scrollTop = chatContainer.scrollHeight;
                activeTyping = setTimeout(step, speed);
            } else {
                el.innerHTML = finalHTML;
                attachPdfLinks(el);
                activeTyping = null;
                done();
            }
        }
        step();

        return {
            stop() {
                clearTimeout(activeTyping);
                activeTyping = null;
            }
        };
    }

    function sendMessage() {
        const msg = userInput.value.trim();
        if (!msg) return;

        appendMessage(msg, "user");
        userInput.value = "";

        // ⬅️ Thinking indicator remains!
        const typingBox = appendMessage("", "bot", true);

        fetch("/ask", {
            method: "POST",
            body: new URLSearchParams({ question: msg }),
            headers: { "Content-Type": "application/x-www-form-urlencoded" }
        })
        .then(res => res.json())
        .then(data => {
            // Remove the thinking bubble
            chatContainer.removeChild(typingBox.parentElement);

            // Add bot’s actual message instantly (no typing animation)
            const botBox = appendMessage("", "bot");
            botBox.innerHTML = marked.parse(data.response);

            attachPdfLinks(botBox);
        })
        .catch(err => {
            appendMessage("Error: " + err.message, "bot");
        });
    }

    sendButton.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", e => {
        if (e.key === "Enter") sendMessage();
    });
});