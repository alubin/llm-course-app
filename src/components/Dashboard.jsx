import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  BookOpen, Clock, CheckCircle2, ArrowRight, 
  Sparkles, Code, Zap, Target
} from 'lucide-react';
import { useProgressContext } from '../hooks/ProgressContext';
import { courseRoadmap } from '../data/courseRoadmap';
import { day1Content } from '../data/day1Content';
import { day2Content } from '../data/day2Content';
import { day3Content } from '../data/day3Content';
import { day4Content } from '../data/day4Content';
import { day5Content } from '../data/day5Content';
import { day6Content } from '../data/day6Content';
import { day7Content } from '../data/day7Content';

export default function Dashboard() {
  const { progress, getCompletedTasksCount } = useProgressContext();

  // Calculate stats across all 7 days
  const allDayContent = [
    day1Content, day2Content, day3Content, day4Content,
    day5Content, day6Content, day7Content
  ];

  const totalTasks = allDayContent.reduce((total, dayContent) => {
    const dayTasks = dayContent.sections
      .filter(s => s.tasks)
      .reduce((acc, s) => acc + s.tasks.length, 0);
    return total + dayTasks;
  }, 0);

  const completedTasks = allDayContent.reduce((total, dayContent, index) => {
    return total + getCompletedTasksCount(index + 1);
  }, 0);

  const progressPercent = Math.round((completedTasks / totalTasks) * 100);

  // Calculate total estimated time
  const totalHoursLow = 4 + 5 + 6 + 6 + 8 + 6 + 6; // 41 hours
  const totalHoursHigh = 8 + 7 + 8 + 8 + 10 + 8 + 8; // 57 hours

  const stats = [
    {
      label: 'Days Available',
      value: '7 / 7',
      icon: BookOpen,
      color: 'text-brand-400'
    },
    {
      label: 'Tasks Completed',
      value: `${completedTasks} / ${totalTasks}`,
      icon: CheckCircle2,
      color: 'text-emerald-400'
    },
    {
      label: 'Est. Time',
      value: `${totalHoursLow}-${totalHoursHigh} hrs`,
      icon: Clock,
      color: 'text-amber-400'
    },
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-surface-900 to-surface-800 border border-surface-700 p-8"
      >
        {/* Background decoration */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-brand-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 left-0 w-48 h-48 bg-emerald-500/10 rounded-full blur-3xl" />
        
        <div className="relative">
          <div className="flex items-center gap-2 text-brand-400 mb-4">
            <Sparkles size={16} />
            <span className="text-sm font-medium">Hands-On Learning Platform</span>
          </div>
          
          <h1 className="text-4xl font-display font-bold text-white mb-4">
            LLM Engineering Course
          </h1>
          
          <p className="text-lg text-surface-300 max-w-2xl mb-6">
            Master Large Language Models through practical, project-based learning. 
            Build real AI applications and create a portfolio of GitHub-ready projects.
          </p>

          <div className="flex flex-wrap gap-4">
            <Link
              to="/day/1"
              className="inline-flex items-center gap-2 px-6 py-3 bg-brand-500 hover:bg-brand-600 text-white font-semibold rounded-lg transition-colors"
            >
              Start Learning
              <ArrowRight size={18} />
            </Link>
            <Link
              to="/python-primer"
              className="inline-flex items-center gap-2 px-6 py-3 bg-surface-700 hover:bg-surface-600 text-white font-semibold rounded-lg transition-colors"
            >
              <Code size={18} />
              Python Primer
            </Link>
          </div>
        </div>
      </motion.div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {stats.map((stat, i) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="bg-surface-900 border border-surface-800 rounded-xl p-6"
          >
            <div className="flex items-center gap-4">
              <div className={`p-3 rounded-lg bg-surface-800 ${stat.color}`}>
                <stat.icon size={24} />
              </div>
              <div>
                <p className="text-2xl font-display font-bold text-white">
                  {stat.value}
                </p>
                <p className="text-sm text-surface-400">{stat.label}</p>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Progress */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="bg-surface-900 border border-surface-800 rounded-xl p-6"
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-display font-semibold text-white">
            Your Progress
          </h2>
          <span className="text-2xl font-bold text-brand-400">
            {progressPercent}%
          </span>
        </div>
        
        <div className="h-3 bg-surface-800 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${progressPercent}%` }}
            transition={{ duration: 1, ease: 'easeOut' }}
            className="h-full bg-gradient-to-r from-brand-500 to-emerald-500 rounded-full"
          />
        </div>
        
        <p className="mt-3 text-sm text-surface-400">
          {completedTasks === 0 
            ? "Start Day 1 to begin tracking your progress"
            : `${completedTasks} of ${totalTasks} tasks completed in Day 1`
          }
        </p>
      </motion.div>

      {/* Current Course */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <h2 className="text-xl font-display font-semibold text-white mb-4">
          Continue Learning
        </h2>
        
        <Link
          to="/day/1"
          className="block bg-surface-900 border border-surface-800 rounded-xl p-6 hover:border-brand-500/50 transition-colors group"
        >
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <span className="px-2 py-1 text-xs font-medium bg-brand-500/20 text-brand-400 rounded">
                  Day 1
                </span>
                <span className="px-2 py-1 text-xs font-medium bg-surface-700 text-surface-300 rounded">
                  Python
                </span>
              </div>
              
              <h3 className="text-xl font-display font-semibold text-white mb-2 group-hover:text-brand-400 transition-colors">
                LLM Fundamentals + CLI Assistant
              </h3>
              
              <p className="text-surface-400 mb-4">
                Learn how LLMs work and build your first AI-powered CLI tool
              </p>
              
              <div className="flex items-center gap-6 text-sm text-surface-500">
                <span className="flex items-center gap-1">
                  <Clock size={14} />
                  4-8 hours
                </span>
                <span className="flex items-center gap-1">
                  <Target size={14} />
                  {totalTasks} tasks
                </span>
                <span className="flex items-center gap-1">
                  <CheckCircle2 size={14} />
                  {completedTasks} completed
                </span>
              </div>
            </div>
            
            <ArrowRight 
              size={24} 
              className="text-surface-600 group-hover:text-brand-400 group-hover:translate-x-1 transition-all" 
            />
          </div>
        </Link>
      </motion.div>

      {/* Features */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <h2 className="text-xl font-display font-semibold text-white mb-4">
          What You'll Learn
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            { icon: Zap, title: 'LLM Fundamentals', desc: 'Tokens, context windows, parameters' },
            { icon: Code, title: 'Python SDKs', desc: 'OpenAI and Anthropic integration' },
            { icon: Target, title: 'CLI Development', desc: 'Build production-ready tools' },
            { icon: Sparkles, title: 'Streaming', desc: 'Real-time response handling' },
          ].map((feature, i) => (
            <div
              key={feature.title}
              className="flex items-start gap-4 p-4 bg-surface-900/50 border border-surface-800 rounded-lg"
            >
              <div className="p-2 bg-surface-800 rounded-lg text-brand-400">
                <feature.icon size={20} />
              </div>
              <div>
                <h3 className="font-semibold text-white">{feature.title}</h3>
                <p className="text-sm text-surface-400">{feature.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
