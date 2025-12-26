import React from 'react';
import { Routes, Route, NavLink, useLocation } from 'react-router-dom';
import {
  BookOpen, Home, Code, Map,
  Terminal, MessageSquare, FileSearch, Server,
  Brain, Bot, Workflow, Menu, X,
  LucideIcon
} from 'lucide-react';
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// Pages
import Dashboard from './components/Dashboard';
import Roadmap from './components/Roadmap';
import CoursePage from './components/CoursePage';
import PythonPrimer from './components/PythonPrimer';

// Context
import { ProgressProvider } from './hooks/ProgressContext';
import { SearchHistoryProvider } from './hooks/SearchHistoryContext';

interface NavItem {
  path: string;
  icon: LucideIcon;
  label: string;
}

function App(): React.ReactElement {
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(false);
  const location = useLocation();

  // Close sidebar on route change (mobile)
  useEffect(() => {
    setSidebarOpen(false);
  }, [location]);

  const navItems: NavItem[] = [
    { path: '/', icon: Home, label: 'Dashboard' },
    { path: '/roadmap', icon: Map, label: 'Course Roadmap' },
    { path: '/python-primer', icon: Code, label: 'Python Primer' },
    { path: '/day/1', icon: Terminal, label: 'Day 1: CLI Assistant' },
    { path: '/day/2', icon: MessageSquare, label: 'Day 2: Memory Chatbot' },
    { path: '/day/3', icon: FileSearch, label: 'Day 3: RAG System' },
    { path: '/day/4', icon: Server, label: 'Day 4: REST API' },
    { path: '/day/5', icon: Brain, label: 'Day 5: Fine-tuning' },
    { path: '/day/6', icon: Bot, label: 'Day 6: AI Agents' },
    { path: '/day/7', icon: Workflow, label: 'Day 7: Data Pipeline' },
  ];


  return (
    <ProgressProvider>
      <SearchHistoryProvider>
        <div className="min-h-screen bg-surface-950 flex">
        {/* Mobile menu button */}
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-surface-800 rounded-lg border border-surface-700"
        >
          {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
        </button>

        {/* Sidebar Overlay (mobile) */}
        <AnimatePresence>
          {sidebarOpen && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden fixed inset-0 bg-black/60 z-40"
            />
          )}
        </AnimatePresence>

        {/* Sidebar */}
        <aside
          className={`
            fixed lg:sticky top-0 left-0 h-screen w-72 bg-surface-900 border-r border-surface-800
            flex flex-col z-40 transition-transform duration-300
            ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
          `}
        >
          {/* Logo */}
          <div className="p-6 border-b border-surface-800">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-500 to-emerald-600 flex items-center justify-center">
                <BookOpen size={20} className="text-white" />
              </div>
              <div>
                <h1 className="font-display font-bold text-white">LLM Course</h1>
                <p className="text-xs text-surface-400">Hands-On Learning</p>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
            {navItems.map(({ path, icon: Icon, label }) => (
              <NavLink
                key={path}
                to={path}
                className={({ isActive }) => `
                  flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200
                  ${isActive
                    ? 'bg-brand-500/10 text-brand-400 border border-brand-500/20'
                    : 'text-surface-400 hover:text-white hover:bg-surface-800'
                  }
                `}
              >
                <Icon size={18} />
                <span className="font-medium">{label}</span>
              </NavLink>
            ))}

          </nav>

        </aside>

        {/* Main Content */}
        <main className="flex-1 min-h-screen lg:ml-0">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:py-12">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/roadmap" element={<Roadmap />} />
              <Route path="/python-primer" element={<PythonPrimer />} />
              <Route path="/day/:dayId" element={<CoursePage />} />
            </Routes>
          </div>
        </main>
        </div>
      </SearchHistoryProvider>
    </ProgressProvider>
  );
}

export default App;
