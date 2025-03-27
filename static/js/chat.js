const chatForm = document.getElementById('chat-form');
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');

    chatForm.addEventListener('submit', async function(event) {
      event.preventDefault();
      const message = userInput.value;
      if (message.trim() === "") return;
      appendMessage('user', message);
      userInput.value = "";

      try {
        const response = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: new URLSearchParams({ query: message })
        });
        const data = await response.json();
        if (data.error) {
          appendMessage('bot', data.error);
        } else {
          // 기본 답변 추가
          appendMessage('bot', data.answer);
          // 만약 pdf_file_name,caption_name이 있다면, 다운로드 링크 추가
          if (data.pdf_file_name !== '' && data.appearance_description !== '' ) {
              console.log("data");
              const downloadUrl = `/download?file=${encodeURIComponent(data.pdf_file_name)}`;
              const linkText = data.caption_name || "다운로드";
              appendDownloadLink(linkText, downloadUrl);
          }

        }
      } catch (error) {
        appendMessage('bot', "서버 오류가 발생했습니다.");
      }
    });

    function appendMessage(sender, text) {
      const messageDiv = document.createElement('div');
      messageDiv.classList.add(sender === 'user' ? 'user-message' : 'bot-message');
      messageDiv.textContent = text;
      chatBox.appendChild(messageDiv);
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    function appendDownloadLink(linkText, url) {
      const linkDiv = document.createElement('div');
      linkDiv.classList.add('bot-message');
      linkDiv.innerHTML = `<a href="${url}" class="download-link" target="_blank">${linkText} 다운로드</a>`;
      chatBox.appendChild(linkDiv);
      chatBox.scrollTop = chatBox.scrollHeight;
    }