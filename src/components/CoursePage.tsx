import React, { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import {
  ChevronRight, ChevronDown, CheckCircle2, Circle,
  Clock, Target, BookOpen, ArrowLeft,
  Copy, Check
} from 'lucide-react';
import { day1Content } from '../data/day1Content';
import { day2Content } from '../data/day2Content';
import { day3Content } from '../data/day3Content';
import { day4Content } from '../data/day4Content';
import { day5Content } from '../data/day5Content';
import { day6Content } from '../data/day6Content';
import { day7Content } from '../data/day7Content';
import { useProgressContext } from '../hooks/ProgressContext';
import { DayContent, Module, Task } from '../data/types';

interface ModuleCardProps {
  module: Module;
  index: number;
  expanded: boolean;
  onToggle: () => void;
}

interface TaskCardProps {
  task: Task;
  index: number;
  dayId: number;
  expanded: boolean;
  completed: boolean;
  onToggleExpand: () => void;
  onToggleComplete: () => void;
}

interface MarkdownContentProps {
  content: string;
}

export default function CoursePage(): React.ReactElement {
  const { dayId } = useParams<{ dayId: string }>();
  const [activeSection, setActiveSection] = useState<string>('theory');
  const [expandedModules, setExpandedModules] = useState<Record<string, boolean>>({});
  const [expandedTasks, setExpandedTasks] = useState<Record<string, boolean>>({});
  const { toggleTask, isTaskCompleted, getCompletedTasksCount } = useProgressContext();

  // Map dayId to content
  const contentMap: Record<string, DayContent> = {
    '1': day1Content,
    '2': day2Content,
    '3': day3Content,
    '4': day4Content,
    '5': day5Content,
    '6': day6Content,
    '7': day7Content
  };

  const content = contentMap[dayId || '1'] || day1Content;

  const totalTasks = content.sections
    .filter(s => s.tasks)
    .reduce((acc, s) => acc + (s.tasks?.length || 0), 0);

  const completedTasks = getCompletedTasksCount(parseInt(dayId || '1'));
  const progressPercent = Math.round((completedTasks / totalTasks) * 100);

  const toggleModule = (moduleId: string): void => {
    setExpandedModules(prev => ({
      ...prev,
      [moduleId]: !prev[moduleId]
    }));
  };

  const toggleTaskExpand = (taskId: string): void => {
    setExpandedTasks(prev => ({
      ...prev,
      [taskId]: !prev[taskId]
    }));
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Link
          to="/roadmap"
          className="inline-flex items-center gap-2 text-surface-400 hover:text-white mb-4 transition-colors"
        >
          <ArrowLeft size={16} />
          Back to Roadmap
        </Link>

        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2 py-1 text-xs font-medium bg-brand-500/20 text-brand-400 rounded">
                Day {dayId}
              </span>
              <span className="px-2 py-1 text-xs font-medium bg-surface-800 text-surface-400 rounded">
                Python
              </span>
            </div>

            <h1 className="text-3xl font-display font-bold text-white mb-2">
              {content.title}
            </h1>

            <p className="text-surface-400 max-w-2xl">
              {content.subtitle}
            </p>
          </div>
        </div>

        {/* Meta */}
        <div className="flex items-center gap-6 mt-4 text-sm text-surface-400">
          <span className="flex items-center gap-1">
            <Clock size={14} />
            {content.duration}
          </span>
          <span className="flex items-center gap-1">
            <Target size={14} />
            {totalTasks} tasks
          </span>
          <span className="flex items-center gap-1">
            <CheckCircle2 size={14} className="text-brand-400" />
            {completedTasks} completed
          </span>
        </div>

        {/* Progress bar */}
        <div className="mt-4">
          <div className="h-2 bg-surface-800 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${progressPercent}%` }}
              transition={{ duration: 0.5 }}
              className="h-full bg-gradient-to-r from-brand-500 to-emerald-500 rounded-full"
            />
          </div>
          <p className="mt-2 text-sm text-surface-500">
            {progressPercent}% complete
          </p>
        </div>
      </motion.div>

      {/* Objectives & Prerequisites */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="grid grid-cols-1 lg:grid-cols-2 gap-6"
      >
        {/* Objectives */}
        <div className="bg-surface-900 border border-surface-800 rounded-xl p-6">
          <h2 className="text-lg font-display font-semibold text-white mb-4 flex items-center gap-2">
            <Target size={18} className="text-brand-400" />
            Learning Objectives
          </h2>
          <ul className="space-y-2">
            {content.objectives.map((obj, i) => (
              <li key={i} className="flex items-start gap-2 text-surface-300">
                <CheckCircle2 size={16} className="text-brand-500 mt-0.5 flex-shrink-0" />
                <span>{obj}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Prerequisites */}
        <div className="bg-surface-900 border border-surface-800 rounded-xl p-6">
          <h2 className="text-lg font-display font-semibold text-white mb-4 flex items-center gap-2">
            <BookOpen size={18} className="text-amber-400" />
            Prerequisites
          </h2>
          <ul className="space-y-3">
            {content.prerequisites.map((prereq, i) => (
              <li key={i} className="flex items-start gap-3">
                <span className="px-2 py-0.5 text-xs font-medium bg-surface-800 text-surface-400 rounded">
                  {prereq.name}
                </span>
                <span className="text-surface-400 text-sm">{prereq.details}</span>
              </li>
            ))}
          </ul>
        </div>
      </motion.div>

      {/* Section Tabs */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="flex gap-2 border-b border-surface-800 pb-4"
      >
        {content.sections.map(section => (
          <button
            key={section.id}
            onClick={() => setActiveSection(section.id)}
            className={`
              px-4 py-2 rounded-lg font-medium transition-all
              ${activeSection === section.id
                ? 'bg-brand-500 text-white'
                : 'bg-surface-800 text-surface-400 hover:text-white'
              }
            `}
          >
            {section.title}
          </button>
        ))}
      </motion.div>

      {/* Content */}
      <AnimatePresence mode="wait">
        {content.sections.map(section => {
          if (section.id !== activeSection) return null;

          return (
            <motion.div
              key={section.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-4"
            >
              <div className="flex items-center gap-4 text-surface-400">
                <Clock size={16} />
                <span>Estimated time: {section.estimatedTime}</span>
              </div>

              {/* Theory Modules */}
              {section.modules && section.modules.map((module, index) => (
                <ModuleCard
                  key={module.id}
                  module={module}
                  index={index}
                  expanded={expandedModules[module.id] ?? index === 0}
                  onToggle={() => toggleModule(module.id)}
                />
              ))}

              {/* Tasks */}
              {section.tasks && section.tasks.map((task, index) => (
                <TaskCard
                  key={task.id}
                  task={task}
                  index={index}
                  dayId={parseInt(dayId || '1')}
                  expanded={expandedTasks[task.id] ?? false}
                  completed={isTaskCompleted(parseInt(dayId || '1'), task.id)}
                  onToggleExpand={() => toggleTaskExpand(task.id)}
                  onToggleComplete={() => toggleTask(parseInt(dayId || '1'), task.id)}
                />
              ))}
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}

function ModuleCard({ module, index, expanded, onToggle }: ModuleCardProps): React.ReactElement {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className="bg-surface-900 border border-surface-800 rounded-xl overflow-hidden"
    >
      <button
        onClick={onToggle}
        className="w-full px-6 py-4 flex items-center justify-between hover:bg-surface-800/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="w-8 h-8 rounded-lg bg-surface-800 flex items-center justify-center text-brand-400 font-mono text-sm">
            {index + 1}
          </span>
          <h3 className="text-lg font-semibold text-white">{module.title}</h3>
        </div>
        {expanded ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="border-t border-surface-800"
          >
            <div className="p-6">
              <MarkdownContent content={module.content} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function TaskCard({
  task,
  index,
  expanded,
  completed,
  onToggleExpand,
  onToggleComplete
}: TaskCardProps): React.ReactElement {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className={`
        bg-surface-900 border rounded-xl overflow-hidden transition-colors
        ${completed ? 'border-brand-500/30' : 'border-surface-800'}
      `}
    >
      <div className="px-6 py-4 flex items-center justify-between">
        <button
          onClick={onToggleExpand}
          className="flex-1 flex items-center gap-3 text-left"
        >
          <span className={`
            w-8 h-8 rounded-lg flex items-center justify-center font-mono text-sm
            ${completed ? 'bg-brand-500/20 text-brand-400' : 'bg-surface-800 text-surface-400'}
          `}>
            {index + 1}
          </span>
          <div>
            <h3 className={`text-lg font-semibold ${completed ? 'text-brand-400' : 'text-white'}`}>
              {task.title}
            </h3>
            <p className="text-sm text-surface-400">{task.description}</p>
          </div>
        </button>

        <div className="flex items-center gap-3">
          <button
            onClick={onToggleComplete}
            className={`
              p-2 rounded-lg transition-all
              ${completed
                ? 'bg-brand-500/20 text-brand-400'
                : 'bg-surface-800 text-surface-500 hover:text-white'
              }
            `}
          >
            {completed ? <CheckCircle2 size={24} /> : <Circle size={24} />}
          </button>

          <button onClick={onToggleExpand}>
            {expanded ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
          </button>
        </div>
      </div>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="border-t border-surface-800"
          >
            <div className="p-6">
              <MarkdownContent content={task.content} />

              {!completed && (
                <button
                  onClick={onToggleComplete}
                  className="mt-6 px-4 py-2 bg-brand-500 hover:bg-brand-600 text-white font-medium rounded-lg transition-colors"
                >
                  Mark as Complete
                </button>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function MarkdownContent({ content }: MarkdownContentProps): React.ReactElement {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyCode = async (code: string, index: string): Promise<void> => {
    await navigator.clipboard.writeText(code);
    setCopiedCode(index);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  // Simple markdown parser
  const parseContent = (text: string): React.ReactElement[] => {
    const lines = text.trim().split('\n');
    const elements: React.ReactElement[] = [];
    let i = 0;
    let codeBlockIndex = 0;

    while (i < lines.length) {
      const line = lines[i];

      // Code block
      if (line.startsWith('```')) {
        const language = line.slice(3).trim() || 'text';
        const codeLines: string[] = [];
        i++;

        while (i < lines.length && !lines[i].startsWith('```')) {
          codeLines.push(lines[i]);
          i++;
        }

        const code = codeLines.join('\n');
        const blockIndex = codeBlockIndex++;

        elements.push(
          <div key={`code-${blockIndex}`} className="relative group my-4">
            <button
              onClick={() => copyCode(code, `code-${blockIndex}`)}
              className="absolute top-2 right-2 p-2 bg-surface-700 hover:bg-surface-600 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity"
            >
              {copiedCode === `code-${blockIndex}` ? (
                <Check size={16} className="text-brand-400" />
              ) : (
                <Copy size={16} className="text-surface-400" />
              )}
            </button>
            <SyntaxHighlighter
              language={language}
              style={oneDark}
              customStyle={{
                margin: 0,
                borderRadius: '0.75rem',
                fontSize: '0.875rem',
              }}
            >
              {code}
            </SyntaxHighlighter>
          </div>
        );
        i++;
        continue;
      }

      // Headers
      if (line.startsWith('### ')) {
        elements.push(
          <h3 key={i} className="text-xl font-display font-semibold text-white mt-6 mb-3">
            {line.slice(4)}
          </h3>
        );
        i++;
        continue;
      }

      if (line.startsWith('## ')) {
        elements.push(
          <h2 key={i} className="text-2xl font-display font-semibold text-white mt-8 mb-4 pb-2 border-b border-surface-800">
            {line.slice(3)}
          </h2>
        );
        i++;
        continue;
      }

      // Blockquote
      if (line.startsWith('> ')) {
        elements.push(
          <blockquote key={i} className="border-l-4 border-brand-500 pl-4 italic text-surface-400 my-4">
            {parseInlineFormatting(line.slice(2))}
          </blockquote>
        );
        i++;
        continue;
      }

      // Table
      if (line.includes('|') && lines[i + 1]?.includes('---')) {
        const tableRows: string[] = [];
        while (i < lines.length && lines[i].includes('|')) {
          tableRows.push(lines[i]);
          i++;
        }

        elements.push(
          <div key={`table-${i}`} className="overflow-x-auto my-4">
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  {tableRows[0].split('|').filter(c => c.trim()).map((cell, j) => (
                    <th key={j} className="bg-surface-800 text-left p-3 font-semibold text-white border border-surface-700">
                      {cell.trim()}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tableRows.slice(2).map((row, j) => (
                  <tr key={j}>
                    {row.split('|').filter(c => c.trim()).map((cell, k) => (
                      <td key={k} className="p-3 border border-surface-800 text-surface-300">
                        {parseInlineFormatting(cell.trim())}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
        continue;
      }

      // List item
      if (line.startsWith('- ')) {
        elements.push(
          <li key={i} className="flex items-start gap-2 text-surface-300 ml-4 my-1">
            <span className="text-brand-500 mt-1">•</span>
            <span>{parseInlineFormatting(line.slice(2))}</span>
          </li>
        );
        i++;
        continue;
      }

      // Numbered list
      if (/^\d+\.\s/.test(line)) {
        const match = line.match(/^(\d+)/);
        const num = match ? match[1] : '';
        const content = line.replace(/^\d+\.\s/, '');
        elements.push(
          <li key={i} className="flex items-start gap-2 text-surface-300 ml-4 my-1">
            <span className="text-brand-500 font-mono">{num}.</span>
            <span>{parseInlineFormatting(content)}</span>
          </li>
        );
        i++;
        continue;
      }

      // Empty line
      if (line.trim() === '') {
        i++;
        continue;
      }

      // Paragraph
      elements.push(
        <p key={i} className="text-surface-300 my-3 leading-relaxed">
          {parseInlineFormatting(line)}
        </p>
      );
      i++;
    }

    return elements;
  };

  const parseInlineFormatting = (text: string): React.ReactNode => {
    // Handle inline code
    const parts = text.split(/(`[^`]+`)/g);

    return parts.map((part, i) => {
      if (part.startsWith('`') && part.endsWith('`')) {
        return (
          <code key={i} className="bg-surface-800 text-brand-400 px-1.5 py-0.5 rounded text-sm font-mono">
            {part.slice(1, -1)}
          </code>
        );
      }

      // Handle bold
      const boldParts = part.split(/(\*\*[^*]+\*\*)/g);
      return boldParts.map((bp, j) => {
        if (bp.startsWith('**') && bp.endsWith('**')) {
          return <strong key={`${i}-${j}`} className="font-semibold text-white">{bp.slice(2, -2)}</strong>;
        }
        return bp;
      });
    });
  };

  return <div className="prose-custom">{parseContent(content)}</div>;
}
