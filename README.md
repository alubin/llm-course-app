# 🎓 LLM Engineering Course

An interactive learning platform for mastering Large Language Model engineering through hands-on projects, featuring an AI-powered Python tutor.

## ✨ Features

- 📚 **Complete 7-Day Curriculum** — All days available with structured learning from fundamentals to advanced topics
- 🤖 **AI-Powered Python Tutor** — Ask questions and get personalized explanations with code examples
- 🔑 **Multi-Provider Support** — Use OpenAI (GPT-4) or Anthropic (Claude) APIs for the AI tutor
- ✅ **Progress Tracking** — Mark tasks complete and track your journey through the course
- 💾 **Persistent Storage** — Progress and search history saved to localStorage
- 📝 **Search History** — Review past AI tutor queries with feedback tracking
- 🐍 **Python Primer** — Comprehensive reference guide for all Python concepts
- 📱 **Responsive Design** — Works seamlessly on desktop, tablet, and mobile
- 🎨 **Beautiful UI** — Dark theme with syntax highlighting and smooth animations

## 🚀 Quick Start

### Prerequisites

- Node.js 18+
- npm or yarn
- (Optional) OpenAI or Anthropic API key for AI Tutor feature

### Installation

```bash
# Clone the repository
git clone https://github.com/alubin/llm-course-app.git
cd llm-course-app

# Install dependencies
npm install

# Start development server
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### Build for Production

```bash
npm run build
npm run preview
```

## 🤖 AI Tutor Setup

The Python Primer includes an AI-powered tutor that can answer any Python questions. To use it:

1. Navigate to **Python Primer** page
2. Click the **AI Tutor** toggle button
3. Click the settings icon to configure your API key
4. Choose your provider:
   - **OpenAI (GPT-4)**: Get an API key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - **Anthropic (Claude)**: Get an API key from [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
5. Enter your API key (stored locally in browser, never sent to our servers)
6. Ask any Python question and get detailed explanations!

### AI Tutor Features

- 📖 **Detailed Explanations** — Conceptual understanding of Python topics
- 💻 **Code Examples** — Working code snippets with syntax highlighting
- ⚠️ **Common Pitfalls** — Learn what to avoid
- 🔗 **LLM Connections** — How each topic relates to LLM development
- 🏷️ **Related Topics** — Discover connected concepts
- 👍 **Feedback System** — Rate responses for better learning
- 📜 **Search History** — Access up to 50 past queries with full responses

## 🌐 Deploy to Vercel

### Option 1: One-Click Deploy

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/alubin/llm-course-app)

### Option 2: Manual Deploy

1. Push your code to GitHub
2. Go to [vercel.com](https://vercel.com)
3. Click "New Project"
4. Import your GitHub repository
5. Click "Deploy"

That's it! Vercel auto-detects Vite projects.

## 📁 Project Structure

```
llm-course-app/
├── public/
│   └── favicon.svg
├── src/
│   ├── components/
│   │   ├── Dashboard.jsx          # Home page with course overview
│   │   ├── Roadmap.jsx            # Visual course roadmap
│   │   ├── CoursePage.jsx         # Individual day content display
│   │   ├── PythonPrimer.jsx       # Python reference + AI tutor
│   │   ├── ApiKeyModal.jsx        # API key configuration modal
│   │   ├── TutorResults.jsx       # AI response display
│   │   ├── SearchHistory.jsx      # Search history sidebar
│   │   └── ProviderBadge.jsx      # OpenAI/Anthropic badges
│   ├── data/
│   │   ├── courseRoadmap.js       # Course structure metadata
│   │   ├── day1Content.js         # Day 1: CLI Assistant
│   │   ├── day2Content.js         # Day 2: Chatbot Memory
│   │   ├── day3Content.js         # Day 3: RAG System
│   │   ├── day4Content.js         # Day 4: REST API (Java)
│   │   ├── day5Content.js         # Day 5: Fine-tuning
│   │   ├── day6Content.js         # Day 6: AI Agents
│   │   ├── day7Content.js         # Day 7: Data Pipeline
│   │   └── pythonPrimer.js        # Python primer content
│   ├── hooks/
│   │   ├── useProgress.js         # Progress tracking hook
│   │   ├── useApiKey.js           # API key management hook
│   │   ├── ProgressContext.jsx    # Global progress context
│   │   └── SearchHistoryContext.jsx # Search history context
│   ├── services/
│   │   └── pythonTutor.js         # OpenAI/Anthropic integration
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
├── vercel.json
└── README.md
```

## 📖 Course Content

| Day | Topic | Tech Stack | Status |
|-----|-------|------------|--------|
| 1 | LLM Fundamentals + CLI Assistant | Python, OpenAI API | ✅ Available |
| 2 | Chatbot with Memory & Context | Python, FastAPI, SQLAlchemy | ✅ Available |
| 3 | RAG: Chat with Documents | Python, ChromaDB, Gradio | ✅ Available |
| 4 | AI-Powered REST API | Java, Spring Boot, Spring AI | ✅ Available |
| 5 | Transformers & Fine-tuning | Python, Hugging Face, LoRA | ✅ Available |
| 6 | AI Agents with Tool Use | Python, Function Calling | ✅ Available |
| 7 | Data Pipeline with AI Enrichment | Python, Pandas, AI Classification | ✅ Available |

### Course Highlights

Each day includes:
- 📝 **Theory Sections** — Core concepts and best practices
- 🛠️ **Hands-On Tasks** — Step-by-step implementation guides
- 💡 **Real-World Applications** — Production-ready patterns
- 🎯 **Learning Objectives** — Clear goals for each day

Total: **41-57 hours** of hands-on learning

## 🛠️ Tech Stack

### Frontend
- **React 18** — UI framework
- **Vite** — Build tool and dev server
- **Tailwind CSS** — Utility-first styling
- **Framer Motion** — Smooth animations
- **React Router** — Client-side routing
- **React Syntax Highlighter** — Code highlighting
- **Lucide React** — Beautiful icons

### AI Integration
- **OpenAI SDK** — GPT-4 integration
- **Anthropic SDK** — Claude integration
- **JSON Mode** — Structured AI responses

### State Management
- **React Context** — Global state (progress, search history)
- **localStorage** — Persistent data storage

## 🎨 Customization

### Adding New Course Content

1. Create a new content file in `src/data/` (e.g., `day8Content.js`)
2. Follow the existing structure with `sections` array
3. Import in `CoursePage.jsx` and add to `contentMap`
4. Update `courseRoadmap.js` with the new day's metadata
5. Add navigation link in `App.jsx`

### Modifying the AI Tutor

The AI tutor service is in `src/services/pythonTutor.js`. You can:
- Customize the system prompt for different teaching styles
- Adjust the response structure in `JSON_SCHEMA`
- Add new AI providers by implementing additional methods
- Modify the model selection (currently uses GPT-4-turbo and Claude-3-sonnet)

### Styling

- **Global styles**: `src/index.css`
- **Tailwind config**: `tailwind.config.js` (custom `surface` and `brand` colors)
- **Component styles**: Inline Tailwind classes

## 🔒 Privacy & Security

- **API keys** are stored locally in browser localStorage
- **No backend** — all AI calls made directly from browser
- **No tracking** — your learning progress stays on your device
- **Open source** — Verify the code yourself

> **Note**: When using `dangerouslyAllowBrowser` flag with AI SDKs, API keys are exposed in browser. Only use for development/learning. Production apps should proxy API calls through a backend.

## 📄 License

MIT License — Feel free to use this for your own learning!

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional course content for advanced topics
- More AI providers (Gemini, Mistral, etc.)
- Backend API proxy for secure API key handling
- Additional language primers (JavaScript, Java, etc.)
- Quizzes and assessments

Please open an issue or PR.

## 🙏 Acknowledgments

- Course content inspired by real-world LLM engineering practices
- Built with modern React and AI best practices
- Designed for hands-on, practical learning

---

Built with ❤️ for aspiring LLM engineers

**🤖 Generated with [Claude Code](https://claude.com/claude-code)**
