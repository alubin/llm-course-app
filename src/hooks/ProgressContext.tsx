import React, { createContext, useContext } from 'react';
import { useProgress, UseProgressReturn } from './useProgress';

const ProgressContext = createContext<UseProgressReturn | null>(null);

export interface ProgressProviderProps {
  children: React.ReactNode;
}

export function ProgressProvider({ children }: ProgressProviderProps): React.ReactElement {
  const progress = useProgress();

  return (
    <ProgressContext.Provider value={progress}>
      {children}
    </ProgressContext.Provider>
  );
}

export function useProgressContext(): UseProgressReturn {
  const context = useContext(ProgressContext);
  if (!context) {
    throw new Error('useProgressContext must be used within a ProgressProvider');
  }
  return context;
}
