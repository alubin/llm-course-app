import { useState, useEffect } from 'react';

const STORAGE_KEY = 'llm-course-progress';

const initialProgress = {
  completedTasks: {},
  completedDays: [],
  currentDay: 1,
  lastVisited: null,
  totalTimeSpent: 0,
};

export function useProgress() {
  const [progress, setProgress] = useState(() => {
    if (typeof window === 'undefined') return initialProgress;
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved ? JSON.parse(saved) : initialProgress;
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
  }, [progress]);

  const toggleTask = (dayId, taskId) => {
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

  const isTaskCompleted = (dayId, taskId) => {
    return (progress.completedTasks[dayId] || []).includes(taskId);
  };

  const getCompletedTasksCount = (dayId) => {
    return (progress.completedTasks[dayId] || []).length;
  };

  const markDayComplete = (dayId) => {
    setProgress(prev => ({
      ...prev,
      completedDays: prev.completedDays.includes(dayId) 
        ? prev.completedDays 
        : [...prev.completedDays, dayId]
    }));
  };

  const isDayComplete = (dayId) => {
    return progress.completedDays.includes(dayId);
  };

  const setCurrentDay = (dayId) => {
    setProgress(prev => ({
      ...prev,
      currentDay: dayId,
      lastVisited: new Date().toISOString()
    }));
  };

  const resetProgress = () => {
    setProgress(initialProgress);
    localStorage.removeItem(STORAGE_KEY);
  };

  const getOverallProgress = (totalTasks) => {
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
