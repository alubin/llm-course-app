import React, { createContext, useContext, useState, useEffect } from 'react';

const SearchHistoryContext = createContext();

const STORAGE_KEY = 'python_tutor_search_history';
const MAX_HISTORY = 50;

export function SearchHistoryProvider({ children }) {
  const [history, setHistory] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        return JSON.parse(stored);
      }
    } catch (error) {
      console.error('Error loading search history:', error);
    }
    return [];
  });

  // Save to localStorage whenever history changes
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    } catch (error) {
      console.error('Error saving search history:', error);
    }
  }, [history]);

  /**
   * Add a new search query and response to history
   */
  const addToHistory = (question, response, helpful = null) => {
    const entry = {
      id: Date.now() + Math.random(), // Unique ID
      question,
      response,
      provider: response.provider,
      model: response.model,
      tokens: response.tokens,
      timestamp: response.timestamp || new Date().toISOString(),
      helpful // User feedback: true, false, or null
    };

    setHistory(prev => {
      const newHistory = [entry, ...prev];
      // Keep only the last MAX_HISTORY entries
      return newHistory.slice(0, MAX_HISTORY);
    });

    return entry.id;
  };

  /**
   * Update feedback on a history entry
   */
  const updateFeedback = (id, helpful) => {
    setHistory(prev =>
      prev.map(entry =>
        entry.id === id ? { ...entry, helpful } : entry
      )
    );
  };

  /**
   * Get a specific history entry by ID
   */
  const getHistoryEntry = (id) => {
    return history.find(entry => entry.id === id);
  };

  /**
   * Clear all history
   */
  const clearHistory = () => {
    setHistory([]);
  };

  /**
   * Get history filtered by provider
   */
  const getHistoryByProvider = (provider) => {
    return history.filter(entry => entry.provider === provider);
  };

  /**
   * Get recent history (last n entries)
   */
  const getRecentHistory = (n = 10) => {
    return history.slice(0, n);
  };

  /**
   * Search history by question text
   */
  const searchHistory = (searchText) => {
    const lowerSearch = searchText.toLowerCase();
    return history.filter(entry =>
      entry.question.toLowerCase().includes(lowerSearch)
    );
  };

  const value = {
    history,
    addToHistory,
    updateFeedback,
    getHistoryEntry,
    clearHistory,
    getHistoryByProvider,
    getRecentHistory,
    searchHistory
  };

  return (
    <SearchHistoryContext.Provider value={value}>
      {children}
    </SearchHistoryContext.Provider>
  );
}

export function useSearchHistory() {
  const context = useContext(SearchHistoryContext);
  if (!context) {
    throw new Error('useSearchHistory must be used within SearchHistoryProvider');
  }
  return context;
}
