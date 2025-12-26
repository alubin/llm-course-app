import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import {
  ChevronRight, ChevronDown, Code, BookOpen,
  Clock, Copy, Check, Search, Sparkles, Settings, History, Loader, AlertCircle
} from 'lucide-react';
import { pythonPrimerContent } from '../data/pythonPrimer';
import { useApiKey } from '../hooks/useApiKey';
import { useSearchHistory } from '../hooks/SearchHistoryContext';
import { pythonTutor } from '../services/pythonTutor';
import ApiKeyModal from './ApiKeyModal';
import TutorResults from './TutorResults';
import SearchHistory from './SearchHistory';
import ProviderBadge from './ProviderBadge';
import { TutorResponse } from '../services/pythonTutor';
import { HistoryEntry } from '../hooks/SearchHistoryContext';
import { PythonSection } from '../data/types';

interface SectionCardProps {
  section: PythonSection;
  index: number;
  expanded: boolean;
  onToggle: () => void;
  copiedCode: string | null;
  onCopyCode: (code: string, index: string) => Promise<void>;
}

interface PrimerContentProps {
  content: string;
  copiedCode: string | null;
  onCopyCode: (code: string, index: string) => Promise<void>;
}

export default function PythonPrimer(): React.ReactElement {
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({ basics: true });
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  // AI Tutor state
  const [isAIMode, setIsAIMode] = useState<boolean>(false);
  const [aiQuestion, setAiQuestion] = useState<string>('');
  const [aiResponse, setAiResponse] = useState<TutorResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [showApiKeyModal, setShowApiKeyModal] = useState<boolean>(false);
  const [showHistory, setShowHistory] = useState<boolean>(false);

  const apiKeyHook = useApiKey();
  const { addToHistory, updateFeedback, history, clearHistory } = useSearchHistory();

  // Auto-show API key modal if no provider is configured (only once when switching to AI mode)
  useEffect(() => {
    if (isAIMode && !apiKeyHook.hasAnyProvider()) {
      setShowApiKeyModal(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAIMode]);

  const handleAISearch = async (e?: React.FormEvent): Promise<void> => {
    e?.preventDefault();
    if (!aiQuestion.trim()) return;

    // Check if provider is configured
    if (!apiKeyHook.hasAnyProvider()) {
      setShowApiKeyModal(true);
      return;
    }

    setIsLoading(true);
    setError(null);
    setAiResponse(null);

    try {
      const provider = apiKeyHook.getActiveProvider();
      if (!provider) {
        throw new Error('No active provider selected');
      }
      const apiKey = apiKeyHook.getApiKey(provider);
      const response = await pythonTutor.ask(aiQuestion, provider, apiKey);

      setAiResponse(response);
      addToHistory(aiQuestion, response);
      setError(null);
    } catch (err) {
      console.error('AI search error:', err);
      setError(err instanceof Error ? err.message : 'Failed to get AI response. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectHistoryQuery = (entry: HistoryEntry): void => {
    setAiQuestion(entry.question);
    setAiResponse(entry.response);
    setIsAIMode(true);
  };

  const handleFeedback = (helpful: boolean): void => {
    if (aiResponse) {
      // Find the history entry with this response
      const entry = history.find(h => h.response.timestamp === aiResponse.timestamp);
      if (entry) {
        updateFeedback(entry.id, helpful);
      }
    }
  };

  const toggleSection = (sectionId: string): void => {
    setExpandedSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }));
  };

  const copyCode = async (code: string, index: string): Promise<void> => {
    await navigator.clipboard.writeText(code);
    setCopiedCode(index);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  const filteredSections = pythonPrimerContent.sections.filter(section =>
    section.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    section.content.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-blue-900/20 to-surface-900 border border-surface-800 p-8"
      >
        <div className="absolute top-0 right-0 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl" />

        <div className="relative">
          <div className="flex items-center gap-2 text-blue-400 mb-4">
            <Code size={16} />
            <span className="text-sm font-medium">Reference Guide</span>
          </div>

          <h1 className="text-3xl font-display font-bold text-white mb-2">
            {pythonPrimerContent.title}
          </h1>

          <p className="text-surface-300 max-w-2xl mb-4">
            {pythonPrimerContent.subtitle}
          </p>

          <div className="flex items-center gap-4 text-sm text-surface-400">
            <span className="flex items-center gap-1">
              <Clock size={14} />
              {pythonPrimerContent.duration}
            </span>
            <span className="flex items-center gap-1">
              <BookOpen size={14} />
              {pythonPrimerContent.sections.length} topics
            </span>
          </div>
        </div>
      </motion.div>

      {/* Search / AI Tutor Toggle */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="space-y-4"
      >
        {/* Mode Toggle */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsAIMode(false)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                !isAIMode
                  ? 'bg-surface-800 text-white'
                  : 'bg-transparent text-surface-400 hover:text-white'
              }`}
            >
              <Search size={16} className="inline mr-2" />
              Browse Topics
            </button>
            <button
              onClick={() => setIsAIMode(true)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                isAIMode
                  ? 'bg-gradient-to-r from-brand-500 to-purple-500 text-white'
                  : 'bg-transparent text-surface-400 hover:text-white'
              }`}
            >
              <Sparkles size={16} className="inline mr-2" />
              AI Tutor
            </button>
          </div>
          {isAIMode && (
            <div className="flex gap-2">
              <button
                onClick={() => setShowHistory(true)}
                className="p-2 bg-surface-800 hover:bg-surface-700 rounded-lg transition-colors"
                title="Search History"
              >
                <History size={18} className="text-surface-400" />
              </button>
              <button
                onClick={() => setShowApiKeyModal(true)}
                className="p-2 bg-surface-800 hover:bg-surface-700 rounded-lg transition-colors"
                title="API Settings"
              >
                <Settings size={18} className="text-surface-400" />
              </button>
            </div>
          )}
        </div>

        {/* Search Input */}
        {!isAIMode ? (
          <div className="relative">
            <Search size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-surface-500" />
            <input
              type="text"
              placeholder="Search topics..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-12 pr-4 py-3 bg-surface-900 border border-surface-800 rounded-xl text-white placeholder-surface-500 focus:outline-none focus:border-brand-500 transition-colors"
            />
          </div>
        ) : (
          <form onSubmit={handleAISearch} className="relative">
            <div className="relative">
              <Sparkles size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-brand-400" />
              <input
                type="text"
                placeholder="Ask me anything about Python..."
                value={aiQuestion}
                onChange={(e) => setAiQuestion(e.target.value)}
                disabled={isLoading}
                className="w-full pl-12 pr-32 py-3 bg-surface-900 border border-surface-800 rounded-xl text-white placeholder-surface-500 focus:outline-none focus:border-brand-500 transition-colors disabled:opacity-50"
              />
              <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-2">
                {apiKeyHook.hasAnyProvider() && (
                  <ProviderBadge provider={apiKeyHook.getActiveProvider()!} size="sm" />
                )}
                <button
                  type="submit"
                  disabled={isLoading || !aiQuestion.trim()}
                  className="px-4 py-1.5 bg-brand-500 hover:bg-brand-600 disabled:bg-surface-700 disabled:text-surface-500 text-white rounded-lg font-medium transition-colors"
                >
                  {isLoading ? (
                    <>
                      <Loader size={14} className="inline mr-1 animate-spin" />
                      Thinking...
                    </>
                  ) : (
                    'Ask'
                  )}
                </button>
              </div>
            </div>
          </form>
        )}

        {/* Error Display */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-start gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-lg"
          >
            <AlertCircle size={18} className="text-red-400 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <p className="text-red-400 text-sm font-medium">Error</p>
              <p className="text-red-300 text-sm mt-1">{error}</p>
            </div>
          </motion.div>
        )}

        {/* AI Results */}
        {isAIMode && aiResponse && (
          <TutorResults
            question={aiQuestion}
            response={aiResponse}
            onFeedback={handleFeedback}
          />
        )}
      </motion.div>

      {/* Quick Navigation */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-surface-900 border border-surface-800 rounded-xl p-6"
      >
        <h2 className="text-lg font-display font-semibold text-white mb-4">
          Quick Navigation
        </h2>
        <div className="flex flex-wrap gap-2">
          {pythonPrimerContent.sections.map(section => (
            <button
              key={section.id}
              onClick={() => {
                setExpandedSections(prev => ({ ...prev, [section.id]: true }));
                document.getElementById(section.id)?.scrollIntoView({ behavior: 'smooth' });
              }}
              className="px-3 py-1.5 text-sm bg-surface-800 hover:bg-surface-700 text-surface-300 hover:text-white rounded-lg transition-colors"
            >
              {section.title}
            </button>
          ))}
        </div>
      </motion.div>

      {/* Sections */}
      <div className="space-y-4">
        {filteredSections.map((section, index) => (
          <SectionCard
            key={section.id}
            section={section}
            index={index}
            expanded={expandedSections[section.id] ?? false}
            onToggle={() => toggleSection(section.id)}
            copiedCode={copiedCode}
            onCopyCode={copyCode}
          />
        ))}
      </div>

      {filteredSections.length === 0 && !isAIMode && (
        <div className="text-center py-12">
          <p className="text-surface-400">No topics found matching "{searchQuery}"</p>
        </div>
      )}

      {/* Modals */}
      <ApiKeyModal
        isOpen={showApiKeyModal}
        onClose={() => setShowApiKeyModal(false)}
        useApiKey={apiKeyHook}
      />

      <SearchHistory
        isOpen={showHistory}
        onClose={() => setShowHistory(false)}
        history={history}
        onSelectQuery={handleSelectHistoryQuery}
        onClearHistory={clearHistory}
      />
    </div>
  );
}

function SectionCard({ section, index, expanded, onToggle, copiedCode, onCopyCode }: SectionCardProps): React.ReactElement {
  return (
    <motion.div
      id={section.id}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.03 }}
      className="bg-surface-900 border border-surface-800 rounded-xl overflow-hidden"
    >
      <button
        onClick={onToggle}
        className="w-full px-6 py-4 flex items-center justify-between hover:bg-surface-800/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center text-blue-400 font-mono text-sm">
            {index + 1}
          </span>
          <h3 className="text-lg font-semibold text-white text-left">{section.title}</h3>
        </div>
        {expanded ? (
          <ChevronDown size={20} className="text-surface-400" />
        ) : (
          <ChevronRight size={20} className="text-surface-400" />
        )}
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="border-t border-surface-800"
          >
            <div className="p-6">
              <PrimerContent
                content={section.content}
                copiedCode={copiedCode}
                onCopyCode={onCopyCode}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function PrimerContent({ content, copiedCode, onCopyCode }: PrimerContentProps): React.ReactElement {
  const parseContent = (text: string): React.ReactElement[] => {
    const lines = text.trim().split('\n');
    const elements: React.ReactElement[] = [];
    let i = 0;
    let codeBlockIndex = 0;

    while (i < lines.length) {
      const line = lines[i];

      // Code block
      if (line.startsWith('```')) {
        const language = line.slice(3).trim() || 'python';
        const codeLines: string[] = [];
        i++;

        while (i < lines.length && !lines[i].startsWith('```')) {
          codeLines.push(lines[i]);
          i++;
        }

        const code = codeLines.join('\n');
        const blockIndex = codeBlockIndex++;

        elements.push(
          <div key={`code-${blockIndex}`} className="relative group my-4">
            <div className="absolute top-2 right-2 flex items-center gap-2">
              <span className="px-2 py-0.5 text-xs bg-surface-700 text-surface-400 rounded">
                {language}
              </span>
              <button
                onClick={() => onCopyCode(code, `primer-${blockIndex}`)}
                className="p-1.5 bg-surface-700 hover:bg-surface-600 rounded opacity-0 group-hover:opacity-100 transition-opacity"
              >
                {copiedCode === `primer-${blockIndex}` ? (
                  <Check size={14} className="text-brand-400" />
                ) : (
                  <Copy size={14} className="text-surface-400" />
                )}
              </button>
            </div>
            <SyntaxHighlighter
              language={language}
              style={oneDark}
              customStyle={{
                margin: 0,
                borderRadius: '0.75rem',
                fontSize: '0.875rem',
              }}
            >
              {code}
            </SyntaxHighlighter>
          </div>
        );
        i++;
        continue;
      }

      // Headers
      if (line.startsWith('### ')) {
        elements.push(
          <h3 key={i} className="text-xl font-display font-semibold text-white mt-6 mb-3">
            {line.slice(4)}
          </h3>
        );
        i++;
        continue;
      }

      if (line.startsWith('## ')) {
        elements.push(
          <h2 key={i} className="text-2xl font-display font-semibold text-white mt-8 mb-4 pb-2 border-b border-surface-800">
            {line.slice(3)}
          </h2>
        );
        i++;
        continue;
      }

      // List item
      if (line.startsWith('- ')) {
        elements.push(
          <li key={i} className="flex items-start gap-2 text-surface-300 ml-4 my-1">
            <span className="text-blue-500 mt-1">•</span>
            <span>{parseInlineFormatting(line.slice(2))}</span>
          </li>
        );
        i++;
        continue;
      }

      // Empty line
      if (line.trim() === '') {
        i++;
        continue;
      }

      // Paragraph
      elements.push(
        <p key={i} className="text-surface-300 my-3 leading-relaxed">
          {parseInlineFormatting(line)}
        </p>
      );
      i++;
    }

    return elements;
  };

  const parseInlineFormatting = (text: string): React.ReactNode => {
    const parts = text.split(/(`[^`]+`)/g);

    return parts.map((part, i) => {
      if (part.startsWith('`') && part.endsWith('`')) {
        return (
          <code key={i} className="bg-surface-800 text-blue-400 px-1.5 py-0.5 rounded text-sm font-mono">
            {part.slice(1, -1)}
          </code>
        );
      }

      const boldParts = part.split(/(\*\*[^*]+\*\*)/g);
      return boldParts.map((bp, j) => {
        if (bp.startsWith('**') && bp.endsWith('**')) {
          return <strong key={`${i}-${j}`} className="font-semibold text-white">{bp.slice(2, -2)}</strong>;
        }
        return bp;
      });
    });
  };

  return <div>{parseContent(content)}</div>;
}
