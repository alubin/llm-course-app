import React from 'react';
import { Sparkles, Zap } from 'lucide-react';

export default function ProviderBadge({ provider, size = 'sm' }) {
  if (!provider) return null;

  const isOpenAI = provider === 'openai';
  const sizeClasses = {
    sm: 'px-2 py-1 text-xs',
    md: 'px-3 py-1.5 text-sm',
    lg: 'px-4 py-2 text-base'
  };

  const iconSize = {
    sm: 12,
    md: 14,
    lg: 16
  };

  return (
    <div className={`inline-flex items-center gap-1.5 rounded-full font-medium ${sizeClasses[size]} ${
      isOpenAI
        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
        : 'bg-orange-500/10 text-orange-400 border border-orange-500/20'
    }`}>
      {isOpenAI ? (
        <Sparkles size={iconSize[size]} />
      ) : (
        <Zap size={iconSize[size]} />
      )}
      <span>
        {isOpenAI ? 'OpenAI' : 'Anthropic'}
      </span>
    </div>
  );
}
