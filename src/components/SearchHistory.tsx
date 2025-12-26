import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Clock, Trash2 } from 'lucide-react';
import ProviderBadge from './ProviderBadge';
import { HistoryEntry } from '../hooks/SearchHistoryContext';

interface SearchHistoryProps {
  isOpen: boolean;
  onClose: () => void;
  history: HistoryEntry[];
  onSelectQuery: (entry: HistoryEntry) => void;
  onClearHistory: () => void;
}

export default function SearchHistory({
  isOpen,
  onClose,
  history,
  onSelectQuery,
  onClearHistory
}: SearchHistoryProps): React.ReactElement | null {
  if (!isOpen) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40 lg:hidden"
          />

          {/* Sidebar */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="fixed top-0 right-0 h-full w-80 bg-surface-900 border-l border-surface-800 shadow-2xl z-50 flex flex-col"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-surface-800">
              <div className="flex items-center gap-2">
                <Clock size={18} className="text-brand-400" />
                <h3 className="font-semibold text-white">Search History</h3>
              </div>
              <button
                onClick={onClose}
                className="p-2 hover:bg-surface-800 rounded-lg transition-colors"
              >
                <X size={18} className="text-surface-400" />
              </button>
            </div>

            {/* History List */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {history.length === 0 ? (
                <div className="text-center py-12">
                  <Clock size={48} className="text-surface-700 mx-auto mb-3" />
                  <p className="text-surface-400 text-sm">No search history yet</p>
                  <p className="text-surface-500 text-xs mt-1">
                    Ask a Python question to get started
                  </p>
                </div>
              ) : (
                history.map((entry) => (
                  <button
                    key={entry.id}
                    onClick={() => {
                      onSelectQuery(entry);
                      onClose();
                    }}
                    className="w-full text-left p-3 bg-surface-800 hover:bg-surface-700 border border-surface-700 rounded-lg transition-colors group"
                  >
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <p className="text-sm text-white line-clamp-2 flex-1">
                        {entry.question}
                      </p>
                      <ProviderBadge provider={entry.provider} size="sm" />
                    </div>
                    <div className="flex items-center gap-2 text-xs text-surface-400">
                      <span>{new Date(entry.timestamp).toLocaleDateString()}</span>
                      <span>•</span>
                      <span>{entry.tokens} tokens</span>
                      {entry.helpful !== null && (
                        <>
                          <span>•</span>
                          <span className={entry.helpful ? 'text-emerald-400' : 'text-red-400'}>
                            {entry.helpful ? '👍' : '👎'}
                          </span>
                        </>
                      )}
                    </div>
                  </button>
                ))
              )}
            </div>

            {/* Footer */}
            {history.length > 0 && (
              <div className="p-4 border-t border-surface-800">
                <button
                  onClick={() => {
                    if (window.confirm('Are you sure you want to clear all history?')) {
                      onClearHistory();
                    }
                  }}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-lg transition-colors"
                >
                  <Trash2 size={16} />
                  <span className="text-sm font-medium">Clear All History</span>
                </button>
              </div>
            )}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
