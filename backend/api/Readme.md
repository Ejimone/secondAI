# Project: OpenCodeAI - A Personal AI Agent

This AI is an AI agent that can get answers from the internet, chat with me with audio and have a conversation with me.
It can also help me with my daily tasks, like sending emails, setting reminders, and more.

It can also help me with my work, like writing reports, creating presentations, and more.
it can do stuffs like getting realtime information from the internet, like weather, news, and more.
It can also help me with my personal life, like planning trips, setting reminders for appointments, and more.
the ai will be like a popup window that i can open and close at any time
it can also be able to do stuffs like:
-when i'm on a website, i can ask it questions, as an agent, it will get answers based on the context of the website
-if i'm on a website, i can ask it to do stuffs like:
-summarize the content of the website
-translate the content of the website
-communicate with audio

I'll be using the below APIs to get answers from the internet:
-Gemini
-OpenAI
-SerpAPI
-AssemblyAI

# Project: OpenCodeAI - A Personal AI Agent

## Core Logic & Architecture

### User Interface (UI)

#### Popup Window

A persistent, lightweight window that can be opened/closed by the user.

- **Contains:**
  - **Text Input:** For user queries and commands.
  - **Output Display:** To show AI responses, information, and summaries.
  - **Audio Input/Output:** For voice commands and responses (integrate with AssemblyAI).
  - **Action Buttons:**
    - "Send" (for text input)
    - "Speak" (for audio input)
    - "Listen" (for audio output)
    - "Close" (to minimize the window)

### AI Engine

#### Core Logic

- **Contextual Understanding:**
  - Analyze user input (text/voice) to understand intent and context.
  - Extract keywords and entities (dates, locations, people, etc.).
  - If on a website:
    - Extract relevant information from the current webpage (using web scraping or browser extensions).
    - Use this context to refine AI responses.
- **Task Execution:**
  - **General Tasks:**
    - **Internet Queries:**
      - Use Gemini or OpenAI for general knowledge, creative content, and complex reasoning.
      - Use SerpAPI for web search and result summarization.
  - **Daily Tasks:**
    - Send emails (integrate with email providers).
    - Set reminders (using calendar APIs or local notifications).
    - Manage schedules and appointments.
  - **Personal Life:**
    - Plan trips (integrate with travel APIs).
    - Find restaurants, events, etc.
  - **Work Tasks:**
    - **Content Creation:**
      - Draft reports, write articles, generate creative content (using Gemini or OpenAI).
      - Create presentations (integrate with presentation software APIs).
    - **Research:**
      - Conduct research on specific topics (using Gemini, OpenAI, and SerpAPI).
      - Summarize research findings.
  - **Website Interactions:**
    - **Summarization:** Use Gemini or OpenAI to summarize the main points of a webpage.
    - **Translation:** Use translation APIs (e.g., Google Translate) to translate the webpage content.
  - **Information Retrieval:**
    - **Real-time Information:**
      - Fetch weather data from weather APIs.
      - Get news updates from news APIs.
      - Track stock prices, cryptocurrency, etc.

### API Integration

- **Gemini:** For general AI tasks, complex reasoning, and creative content generation.
- **OpenAI:** For similar capabilities as Gemini, with potential for different strengths/weaknesses.
- **SerpAPI:** For web search, result summarization, and extracting structured data from search results.
- **AssemblyAI:** For real-time speech-to-text and text-to-speech conversion.
- **Email APIs:** For sending and receiving emails (e.g., Gmail API, Outlook API).
- **Calendar APIs:** For managing schedules and setting reminders.
- **Travel APIs:** For booking flights, hotels, etc.
- **Weather APIs:** For fetching current weather conditions and forecasts.
- **News APIs:** For accessing news articles and headlines.

### Development Considerations

- **UI/UX:** Prioritize a clean, intuitive, and user-friendly interface.
- **Performance:** Optimize for speed and responsiveness to ensure a smooth user experience.
- **Security:** Implement robust security measures to protect user data and API keys.
- **Privacy:** Adhere to user privacy regulations (e.g., GDPR, CCPA).
- **Scalability:** Design the system to handle increasing user demands and data volumes.

### Key Features

- **Contextual Awareness:** Understands user queries and the current context (website, user location, etc.).
- **Multi-modal Interaction:** Supports both text and voice input/output.
- **Proactive Assistance:** Can anticipate user needs and offer relevant suggestions.
- **Personalization:** Learns user preferences and adapts to individual needs.
- **Continuous Improvement:** Regularly updates and improves its capabilities based on user feedback and new technologies.

---

## Core Components

1. **User Interaction Module**

   - **Popup Window Interface:** A lightweight UI that can open/close at any time.
   - **Input Modes:** Text input, voice input (via AssemblyAI), and buttons for common tasks.
   - **Output Modes:** Text display, audio output (via AssemblyAI).

2. **AI Engine**

   - **Natural Language Processing:** Powered by OpenAI for understanding and generating text.
   - **Real-Time Context Analysis:** Processes website content or user context dynamically.
   - **Task Automation:** Handles commands like sending emails, setting reminders, and performing actions.

3. **API Integration**

   - **Gemini API:** High-level AI interactions and advanced contextual reasoning.
   - **OpenAI API:** General conversational abilities and content generation.
   - **SerpAPI:** Fetches internet-based data like weather, news, and other real-time information.
   - **AssemblyAI:** Transcribes voice inputs and converts AI responses to audio.

4. **Task Handler**

   - **Daily Tasks:** Schedules reminders, sends emails, manages calendar events.
   - **Work Assistance:** Generates reports, presentations, and other professional documents.
   - **Personal Life:** Plans trips, manages appointments, and fetches location-based details.

5. **Web Context Module**

   - **Context Extraction:** Scrapes or gathers the current page’s content for analysis (using SerpAPI or custom web scraping).
   - **Contextual Actions:**
     - Summarize content.
     - Translate content.
     - Interact with content contextually.
   - **Audio Interaction:** Enables seamless audio-based engagement with web content.

6. **Real-Time Information Fetcher**
   - Fetches data like weather, news, stock updates, and more using SerpAPI.

---

## Logic Workflow

### 1. Popup Window Initialization

- User opens the popup.
- Selects the interaction type: Text/Voice/Command.
- Interface connects to the backend for processing.

### 2. Command Processing

- **Input Type Check:**
  - **Text:** Sent directly to the AI Engine.
  - **Voice:** Transcribed via AssemblyAI and or google-speech-to-text forwarded to the AI Engine.
- **Command Classification:**
  - Chat-based queries: Use OpenAI/Gemini.
  - Real-time data: Use SerpAPI.
  - Personal/Work tasks: Route to Task Handler.

### 3. Web Context Interaction

- Detect active web page and extract content (using browser extensions or plugins).
- User provides a command:
  - **Summarization:** Use OpenAI/Gemini to summarize.
  - **Translation:** Use OpenAI to translate.
  - **Audio Interaction:** Convert outputs to audio with AssemblyAI.

### 4. Task Automation

- Email: Integrated with your email service (e.g., Gmail API).
- Reminders: Use a calendar API (Google Calendar or a local storage solution).
- Reports/Presentations: Generate structured content using OpenAI/Gemini.

### 5. Real-Time Information

- Fetch news, weather, or live updates from SerpAPI.
- Parse and present the results to the user.

### 6. Response Handling

- Text: Directly display the response in the popup.
- Voice: Convert to audio using AssemblyAI and play it in the popup.

---

## Data Flow

1. **User Input → NLP Engine**

   - Text → OpenAI/Gemini.
   - Voice → AssemblyAI → OpenAI/Gemini.

2. **Task Request → Task Handler**

   - Command classification → Relevant handler (Daily/Work/Personal).

3. **Web Interaction → Context Module**

   - Fetch website data → Process commands like summarize, translate.

-

4. **Real-Time Queries → API Integration**

   - Weather/news → SerpAPI.

5. **Output Delivery**
   - Text → Display in popup.
   - Voice → AssemblyAI for audio response.

---

## Technology Stack

- **Frontend:** html, css, and js
- **Backend:** Python (FastAPI/Django) to handle API integration and logic.
- **Database:** PostgreSQL or MongoDB for user data, reminders, and tasks.
- **APIs:** Gemini, OpenAI, SerpAPI, AssemblyAI.
- **Browser Extension:** For seamless web content interaction.

---

## Development Phases

1. **Phase 1:** Develop core popup functionality with basic NLP using OpenAI.
2. **Phase 2:** Integrate voice capabilities via AssemblyAI.
3. **Phase 3:** Implement task automation (email, reminders).
4. **Phase 4:** Add real-time data fetching (weather, news).
5. **Phase 5:** Build the web context interaction module.
6. **Phase 6:** Optimize and refine for production.

---

## Chrome Extension Integration

### Overview

The SecondAI assistant will also be available as a Chrome extension. Once activated, it will appear as a popup on the screen and perform actions seamlessly.

### Extension Structure

The extension will have the following structure:

- **Popup Window:** The main interface for the user to interact with the assistant.
- **Background Script:** Handles communication between the popup and the backend.
- **Content Script:** Handles the interaction with the web page and performs actions based on user commands.

### Technology Stack

- **Frontend:** HTML, CSS, JavaScript for the popup window.
- **Backend:** Flask to handle API integration and logic.
- **Database:** PostgreSQL or MongoDB for user data, reminders, and tasks.
- **APIs:** Gemini, OpenAI, SerpAPI, AssemblyAI.
- **Browser Extension:** For seamless web content interaction.

---

#importing the modules from the other files, here, the bot will be able to use the functions from the other files, it will listen to the wakeword, if a command is given it would analyse the prompt, check for the right functions to call and assign that to the neccessaty function, the bot will be able to send emails, create tasks, summarize content, scrape webpages, get the weather, search in real time, process documents, ask questions, stream answers, stream web search, analyze prompts and route tasks, manage todos, and send emails, else, it would answer the default system message, and the user can end the conversation by saying "thank you for your help", the bot will respond with "You're welcome! Have a great day!" and the conversation will end, the bot will be able to respond to any question or statement that is asked of it or tasked to it, it will generate words in a user-friendly manner, it can also ask questions to the user to get more information, be playful and generate words of value prioritising logic and facts

# from ai import send_email, handle_task_creation, create_task, get_gmail_service, scrape_url, summarize_content

# from docsprocessing import RAGProcessor # Import the RAGProcessor class from docsprocessing.py, RAGPprocessor it has functions to process_documentsm \_process_urls, \_process_pdf, ask_question, stream_answer, \_stram_web_search and the main function which was used to test the class, which could be modified to be used to be used

# from realtimeSearch import get_current_time, real_time_search

# from weather import get_weather

# from sendEmail import test_service, send_email

# from tasks import analyze_prompt_and_route_task

# from todo import TodoManager

# from webScrapeAndProcess import web_search, scrape_url, summarize_content, scrape_webpages_with_serpapi
