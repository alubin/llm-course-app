import React, { createContext, useContext, useState, useEffect } from 'react';
import { TutorResponse, AIProvider } from '../services/pythonTutor';

const SearchHistoryContext = createContext<SearchHistoryContextValue | null>(null);

const STORAGE_KEY = 'python_tutor_search_history';
const MAX_HISTORY = 50;

export interface HistoryEntry {
  id: number;
  question: string;
  response: TutorResponse;
  provider: AIProvider;
  model: string;
  tokens: number;
  timestamp: string;
  helpful: boolean | null;
}

export interface SearchHistoryContextValue {
  history: HistoryEntry[];
  addToHistory: (question: string, response: TutorResponse, helpful?: boolean | null) => number;
  updateFeedback: (id: number, helpful: boolean) => void;
  getHistoryEntry: (id: number) => HistoryEntry | undefined;
  clearHistory: () => void;
  getHistoryByProvider: (provider: AIProvider) => HistoryEntry[];
  getRecentHistory: (n?: number) => HistoryEntry[];
  searchHistory: (searchText: string) => HistoryEntry[];
}

export interface SearchHistoryProviderProps {
  children: React.ReactNode;
}

export function SearchHistoryProvider({ children }: SearchHistoryProviderProps): React.ReactElement {
  const [history, setHistory] = useState<HistoryEntry[]>(() => {
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
  const addToHistory = (question: string, response: TutorResponse, helpful: boolean | null = null): number => {
    const entry: HistoryEntry = {
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
  const updateFeedback = (id: number, helpful: boolean): void => {
    setHistory(prev =>
      prev.map(entry =>
        entry.id === id ? { ...entry, helpful } : entry
      )
    );
  };

  /**
   * Get a specific history entry by ID
   */
  const getHistoryEntry = (id: number): HistoryEntry | undefined => {
    return history.find(entry => entry.id === id);
  };

  /**
   * Clear all history
   */
  const clearHistory = (): void => {
    setHistory([]);
  };

  /**
   * Get history filtered by provider
   */
  const getHistoryByProvider = (provider: AIProvider): HistoryEntry[] => {
    return history.filter(entry => entry.provider === provider);
  };

  /**
   * Get recent history (last n entries)
   */
  const getRecentHistory = (n: number = 10): HistoryEntry[] => {
    return history.slice(0, n);
  };

  /**
   * Search history by question text
   */
  const searchHistory = (searchText: string): HistoryEntry[] => {
    const lowerSearch = searchText.toLowerCase();
    return history.filter(entry =>
      entry.question.toLowerCase().includes(lowerSearch)
    );
  };

  const value: SearchHistoryContextValue = {
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

export function useSearchHistory(): SearchHistoryContextValue {
  const context = useContext(SearchHistoryContext);
  if (!context) {
    throw new Error('useSearchHistory must be used within SearchHistoryProvider');
  }
  return context;
}
