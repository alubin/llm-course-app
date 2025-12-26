# 🎓 LLM Engineering Course

An interactive learning platform for mastering Large Language Model engineering through hands-on projects.

![Course Preview](https://via.placeholder.com/800x400/18181b/22c55e?text=LLM+Engineering+Course)

## ✨ Features

- 📚 **7-Day Curriculum** — Structured learning path from fundamentals to advanced topics
- ✅ **Progress Tracking** — Mark tasks complete and track your journey
- 💾 **Persistent Storage** — Progress saved to localStorage
- 🐍 **Python Primer** — Reference guide for all Python concepts used
- 📱 **Responsive Design** — Works on desktop, tablet, and mobile
- 🎨 **Beautiful UI** — Dark theme with syntax highlighting

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-engineering-course.git
cd llm-engineering-course

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

## 🌐 Deploy to Vercel

### Option 1: One-Click Deploy

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/yourusername/llm-engineering-course)

### Option 2: Manual Deploy

1. Push your code to GitHub
2. Go to [vercel.com](https://vercel.com)
3. Click "New Project"
4. Import your GitHub repository
5. Click "Deploy"

That's it! Vercel auto-detects Vite projects.

## 📁 Project Structure

```
llm-engineering-course/
├── public/
│   └── favicon.svg
├── src/
│   ├── components/
│   │   ├── Dashboard.jsx      # Home page with overview
│   │   ├── Roadmap.jsx        # Course roadmap view
│   │   ├── CoursePage.jsx     # Individual day content
│   │   └── PythonPrimer.jsx   # Python reference guide
│   ├── data/
│   │   ├── courseRoadmap.js   # Course structure
│   │   ├── day1Content.js     # Day 1 full content
│   │   └── pythonPrimer.js    # Python primer content
│   ├── hooks/
│   │   ├── useProgress.js     # Progress tracking hook
│   │   └── ProgressContext.jsx
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
└── README.md
```

## 📖 Course Content

| Day | Topic | Status |
|-----|-------|--------|
| 1 | LLM Fundamentals + CLI Assistant | ✅ Available |
| 2 | Chatbot with Memory & Context | 🔜 Coming Soon |
| 3 | RAG: Chat with Documents | 🔜 Coming Soon |
| 4 | AI-Powered REST API (Java) | 🔜 Coming Soon |
| 5 | Transformers & Fine-tuning | 🔜 Coming Soon |
| 6 | AI Agents with Tool Use | 🔜 Coming Soon |
| 7 | Data Pipeline with AI | 🔜 Coming Soon |

## 🛠️ Tech Stack

- **React 18** — UI framework
- **Vite** — Build tool
- **Tailwind CSS** — Styling
- **Framer Motion** — Animations
- **React Router** — Navigation
- **React Syntax Highlighter** — Code blocks
- **Lucide React** — Icons

## 🎨 Customization

### Adding New Course Days

1. Create content file in `src/data/day2Content.js`
2. Import in `CoursePage.jsx`
3. Update routing logic to load appropriate content

### Modifying Styles

- Global styles: `src/index.css`
- Tailwind config: `tailwind.config.js`
- Component-specific: Inline Tailwind classes

## 📄 License

MIT License — Feel free to use this for your own learning!

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

---

Built with ❤️ for aspiring LLM engineers
