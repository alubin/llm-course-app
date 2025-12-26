import React from 'react';
import { Sparkles, Zap, Cpu, LucideIcon } from 'lucide-react';
import { AIProvider } from '../services/pythonTutor';

type BadgeSize = 'sm' | 'md' | 'lg';

interface ProviderBadgeProps {
  provider: AIProvider;
  size?: BadgeSize;
}

interface ProviderConfig {
  icon: LucideIcon;
  label: string;
  className: string;
}

export default function ProviderBadge({ provider, size = 'sm' }: ProviderBadgeProps): React.ReactElement | null {
  if (!provider) return null;

  const sizeClasses: Record<BadgeSize, string> = {
    sm: 'px-2 py-1 text-xs',
    md: 'px-3 py-1.5 text-sm',
    lg: 'px-4 py-2 text-base'
  };

  const iconSize: Record<BadgeSize, number> = {
    sm: 12,
    md: 14,
    lg: 16
  };

  const getProviderConfig = (): ProviderConfig => {
    switch (provider) {
      case 'openai':
        return {
          icon: Sparkles,
          label: 'OpenAI',
          className: 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
        };
      case 'anthropic':
        return {
          icon: Zap,
          label: 'Anthropic',
          className: 'bg-orange-500/10 text-orange-400 border border-orange-500/20'
        };
      case 'gemini':
        return {
          icon: Cpu,
          label: 'Gemini',
          className: 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
        };
      default:
        return {
          icon: Sparkles,
          label: 'Unknown',
          className: 'bg-surface-500/10 text-surface-400 border border-surface-500/20'
        };
    }
  };

  const config = getProviderConfig();
  const Icon = config.icon;

  return (
    <div className={`inline-flex items-center gap-1.5 rounded-full font-medium ${sizeClasses[size]} ${config.className}`}>
      <Icon size={iconSize[size]} />
      <span>{config.label}</span>
    </div>
  );
}
