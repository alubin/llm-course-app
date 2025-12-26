import { useState, useEffect } from 'react';

const STORAGE_KEY = 'llm-course-progress';

export interface Progress {
  completedTasks: Record<number, string[]>;
  completedDays: number[];
  currentDay: number;
  lastVisited: string | null;
  totalTimeSpent: number;
}

export interface UseProgressReturn {
  progress: Progress;
  toggleTask: (dayId: number, taskId: string) => void;
  isTaskCompleted: (dayId: number, taskId: string) => boolean;
  getCompletedTasksCount: (dayId: number) => number;
  markDayComplete: (dayId: number) => void;
  isDayComplete: (dayId: number) => boolean;
  setCurrentDay: (dayId: number) => void;
  resetProgress: () => void;
  getOverallProgress: (totalTasks: number) => number;
}

const initialProgress: Progress = {
  completedTasks: {},
  completedDays: [],
  currentDay: 1,
  lastVisited: null,
  totalTimeSpent: 0,
};

export function useProgress(): UseProgressReturn {
  const [progress, setProgress] = useState<Progress>(() => {
    if (typeof window === 'undefined') return initialProgress;
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved ? JSON.parse(saved) : initialProgress;
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
  }, [progress]);

  const toggleTask = (dayId: number, taskId: string): void => {
    setProgress(prev => {
      const dayTasks = prev.completedTasks[dayId] || [];
      const isCompleted = dayTasks.includes(taskId);

      return {
        ...prev,
        completedTasks: {
          ...prev.completedTasks,
          [dayId]: isCompleted
            ? dayTasks.filter(t => t !== taskId)
            : [...dayTasks, taskId]
        }
      };
    });
  };

  const isTaskCompleted = (dayId: number, taskId: string): boolean => {
    return (progress.completedTasks[dayId] || []).includes(taskId);
  };

  const getCompletedTasksCount = (dayId: number): number => {
    return (progress.completedTasks[dayId] || []).length;
  };

  const markDayComplete = (dayId: number): void => {
    setProgress(prev => ({
      ...prev,
      completedDays: prev.completedDays.includes(dayId)
        ? prev.completedDays
        : [...prev.completedDays, dayId]
    }));
  };

  const isDayComplete = (dayId: number): boolean => {
    return progress.completedDays.includes(dayId);
  };

  const setCurrentDay = (dayId: number): void => {
    setProgress(prev => ({
      ...prev,
      currentDay: dayId,
      lastVisited: new Date().toISOString()
    }));
  };

  const resetProgress = (): void => {
    setProgress(initialProgress);
    localStorage.removeItem(STORAGE_KEY);
  };

  const getOverallProgress = (totalTasks: number): number => {
    const completed = Object.values(progress.completedTasks).flat().length;
    return Math.round((completed / totalTasks) * 100);
  };

  return {
    progress,
    toggleTask,
    isTaskCompleted,
    getCompletedTasksCount,
    markDayComplete,
    isDayComplete,
    setCurrentDay,
    resetProgress,
    getOverallProgress,
  };
}
