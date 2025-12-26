import React, { createContext, useContext } from 'react';
import { useProgress } from './useProgress';

const ProgressContext = createContext(null);

export function ProgressProvider({ children }) {
  const progress = useProgress();
  
  return (
    <ProgressContext.Provider value={progress}>
      {children}
    </ProgressContext.Provider>
  );
}

export function useProgressContext() {
  const context = useContext(ProgressContext);
  if (!context) {
    throw new Error('useProgressContext must be used within a ProgressProvider');
  }
  return context;
}
