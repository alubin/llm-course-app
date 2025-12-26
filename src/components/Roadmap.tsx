import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Terminal, MessageSquare, FileSearch, Server,
  Brain, Bot, Workflow, Clock, Lock, ArrowRight,
  CheckCircle2, LucideIcon
} from 'lucide-react';
import { courseRoadmap } from '../data/courseRoadmap';
import { useProgressContext } from '../hooks/ProgressContext';
import { CourseDay } from '../data/types';

const iconMap: Record<string, LucideIcon> = {
  Terminal,
  MessageSquare,
  FileSearch,
  Server,
  Brain,
  Bot,
  Workflow,
};

interface DayContentProps {
  day: CourseDay;
  isAvailable: boolean;
}

export default function Roadmap(): React.ReactElement {
  const { isDayComplete } = useProgressContext();

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-3xl font-display font-bold text-white mb-2">
          Course Roadmap
        </h1>
        <p className="text-surface-400 max-w-2xl">
          Your 7-day journey to mastering LLM engineering. Each day focuses on
          building a real, portfolio-ready project.
        </p>
      </motion.div>

      {/* Timeline */}
      <div className="relative">
        {/* Vertical line */}
        <div className="absolute left-6 top-0 bottom-0 w-px bg-surface-800" />

        <div className="space-y-6">
          {courseRoadmap.map((day, index) => {
            const Icon = iconMap[day.icon] || Terminal;
            const isAvailable = day.status === 'available';
            const isComplete = isDayComplete(day.id);

            return (
              <motion.div
                key={day.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`relative pl-16 ${!isAvailable ? 'opacity-60' : ''}`}
              >
                {/* Circle indicator */}
                <div
                  className={`
                    absolute left-0 w-12 h-12 rounded-full flex items-center justify-center
                    ${isComplete
                      ? 'bg-brand-500 text-white'
                      : isAvailable
                        ? 'bg-surface-800 text-brand-400 border-2 border-brand-500'
                        : 'bg-surface-800 text-surface-500 border border-surface-700'
                    }
                  `}
                >
                  {isComplete ? (
                    <CheckCircle2 size={24} />
                  ) : (
                    <Icon size={24} />
                  )}
                </div>

                {/* Card */}
                <div
                  className={`
                    bg-surface-900 border rounded-xl p-6 transition-all
                    ${isAvailable
                      ? 'border-surface-800 hover:border-brand-500/50 cursor-pointer'
                      : 'border-surface-800/50'
                    }
                  `}
                >
                  {isAvailable ? (
                    <Link to={`/day/${day.id}`} className="block">
                      <DayContent day={day} isAvailable={isAvailable} />
                    </Link>
                  ) : (
                    <DayContent day={day} isAvailable={isAvailable} />
                  )}
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Legend */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="flex items-center gap-6 text-sm text-surface-400 pt-4"
      >
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-brand-500" />
          <span>Available</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-surface-700 border border-surface-600" />
          <span>Coming Soon</span>
        </div>
        <div className="flex items-center gap-2">
          <CheckCircle2 size={14} className="text-brand-400" />
          <span>Completed</span>
        </div>
      </motion.div>
    </div>
  );
}

function DayContent({ day, isAvailable }: DayContentProps): React.ReactElement {
  return (
    <>
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span className={`
              px-2 py-1 text-xs font-medium rounded
              ${isAvailable
                ? 'bg-brand-500/20 text-brand-400'
                : 'bg-surface-800 text-surface-500'
              }
            `}>
              Day {day.id}
            </span>
            <span className="px-2 py-1 text-xs font-medium bg-surface-800 text-surface-400 rounded">
              {day.language}
            </span>
            {!isAvailable && (
              <span className="flex items-center gap-1 px-2 py-1 text-xs font-medium bg-surface-800 text-surface-500 rounded">
                <Lock size={10} />
                Coming Soon
              </span>
            )}
          </div>

          <h3 className={`
            text-xl font-display font-semibold mb-2
            ${isAvailable ? 'text-white' : 'text-surface-400'}
          `}>
            {day.title}
          </h3>

          <p className="text-surface-400 mb-4">
            {day.description}
          </p>
        </div>

        {isAvailable && (
          <ArrowRight
            size={20}
            className="text-surface-600 group-hover:text-brand-400 flex-shrink-0"
          />
        )}
      </div>

      {/* Topics */}
      <div className="flex flex-wrap gap-2 mb-4">
        {day.topics.slice(0, 4).map((topic: string) => (
          <span
            key={topic}
            className="px-2 py-1 text-xs bg-surface-800 text-surface-400 rounded"
          >
            {topic}
          </span>
        ))}
        {day.topics.length > 4 && (
          <span className="px-2 py-1 text-xs bg-surface-800 text-surface-500 rounded">
            +{day.topics.length - 4} more
          </span>
        )}
      </div>

      {/* Meta */}
      <div className="flex items-center gap-4 text-sm text-surface-500">
        <span className="flex items-center gap-1">
          <Clock size={14} />
          {day.duration}
        </span>
        <span>
          Project: {day.project}
        </span>
      </div>
    </>
  );
}
