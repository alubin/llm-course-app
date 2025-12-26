import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check, ThumbsUp, ThumbsDown, Lightbulb, AlertCircle, Sparkles } from 'lucide-react';
import ProviderBadge from './ProviderBadge';

export default function TutorResults({ question, response, onFeedback }) {
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState(null); // true = helpful, false = not helpful

  const handleCopy = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleFeedback = (helpful) => {
    setFeedback(helpful);
    onFeedback?.(helpful);
  };

  if (!response) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="mt-6 space-y-4"
    >
      {/* Question */}
      <div className="bg-surface-800/50 border border-surface-700 rounded-lg p-4">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            <p className="text-sm text-surface-400 mb-1">Your question</p>
            <p className="text-white font-medium">{question}</p>
          </div>
          <ProviderBadge provider={response.provider} />
        </div>
      </div>

      {/* Response */}
      <div className="bg-surface-900 border border-surface-700 rounded-lg overflow-hidden">
        {/* Explanation */}
        {response.explanation && (
          <div className="p-6 border-b border-surface-800">
            <div className="flex items-center gap-2 mb-3">
              <Sparkles size={18} className="text-brand-400" />
              <h4 className="text-white font-semibold">Explanation</h4>
            </div>
            <p className="text-surface-200 leading-relaxed">
              {response.explanation}
            </p>
          </div>
        )}

        {/* Code Example */}
        {response.code && (
          <div className="border-b border-surface-800">
            <div className="flex items-center justify-between px-6 py-3 bg-surface-800/50">
              <span className="text-sm text-surface-400 font-medium">Code Example</span>
              <button
                onClick={() => handleCopy(response.code)}
                className="flex items-center gap-2 px-3 py-1.5 text-sm bg-surface-700 hover:bg-surface-600 text-surface-300 hover:text-white rounded-md transition-colors"
              >
                {copied ? (
                  <>
                    <Check size={14} />
                    <span>Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy size={14} />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
            <div className="overflow-x-auto">
              <SyntaxHighlighter
                language="python"
                style={oneDark}
                customStyle={{
                  margin: 0,
                  padding: '1.5rem',
                  background: '#1a1a1f',
                  fontSize: '14px'
                }}
              >
                {response.code}
              </SyntaxHighlighter>
            </div>
          </div>
        )}

        {/* Pitfalls */}
        {response.pitfalls && (
          <div className="p-6 border-b border-surface-800">
            <div className="flex items-start gap-3">
              <AlertCircle size={18} className="text-amber-400 mt-0.5 flex-shrink-0" />
              <div>
                <h4 className="text-white font-semibold mb-2">Common Pitfalls</h4>
                <p className="text-surface-200 leading-relaxed">
                  {response.pitfalls}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* LLM Connection */}
        {response.llm_connection && (
          <div className="p-6 border-b border-surface-800 bg-brand-500/5">
            <div className="flex items-start gap-3">
              <Lightbulb size={18} className="text-brand-400 mt-0.5 flex-shrink-0" />
              <div>
                <h4 className="text-white font-semibold mb-2">Connection to LLM Development</h4>
                <p className="text-surface-200 leading-relaxed">
                  {response.llm_connection}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Related Topics */}
        {response.related_topics && response.related_topics.length > 0 && (
          <div className="p-6 border-b border-surface-800">
            <h4 className="text-sm text-surface-400 mb-3">Related Topics</h4>
            <div className="flex flex-wrap gap-2">
              {response.related_topics.map((topic, index) => (
                <span
                  key={index}
                  className="px-3 py-1.5 bg-surface-800 border border-surface-700 text-surface-300 text-sm rounded-lg"
                >
                  {topic}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Feedback */}
        <div className="p-4 bg-surface-800/30">
          <div className="flex items-center justify-between">
            <p className="text-sm text-surface-400">Was this explanation helpful?</p>
            <div className="flex gap-2">
              <button
                onClick={() => handleFeedback(true)}
                className={`p-2 rounded-lg transition-colors ${
                  feedback === true
                    ? 'bg-emerald-500/20 text-emerald-400'
                    : 'bg-surface-800 text-surface-400 hover:bg-surface-700 hover:text-white'
                }`}
              >
                <ThumbsUp size={16} />
              </button>
              <button
                onClick={() => handleFeedback(false)}
                className={`p-2 rounded-lg transition-colors ${
                  feedback === false
                    ? 'bg-red-500/20 text-red-400'
                    : 'bg-surface-800 text-surface-400 hover:bg-surface-700 hover:text-white'
                }`}
              >
                <ThumbsDown size={16} />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Metadata */}
      <div className="flex items-center gap-4 text-xs text-surface-500">
        <span>Model: {response.model}</span>
        <span>•</span>
        <span>Tokens: {response.tokens}</span>
        <span>•</span>
        <span>{new Date(response.timestamp).toLocaleString()}</span>
      </div>
    </motion.div>
  );
}
