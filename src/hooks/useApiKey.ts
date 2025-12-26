import { useState, useEffect } from 'react';

const API_KEY_STORAGE_KEY = 'llm_course_api_keys';

export const PROVIDERS = {
  OPENAI: 'openai',
  ANTHROPIC: 'anthropic',
  GEMINI: 'gemini'
} as const;

export type ProviderType = typeof PROVIDERS[keyof typeof PROVIDERS];

export interface ApiKeysState {
  openai: string;
  anthropic: string;
  gemini: string;
  activeProvider: ProviderType | null;
  lastUpdated: string | null;
}

export interface UseApiKeyReturn {
  // Getters
  getApiKey: (provider: ProviderType) => string;
  getActiveProvider: () => ProviderType | null;
  getConfiguredProviders: () => ProviderType[];
  isProviderConfigured: (provider: ProviderType) => boolean;
  hasAnyProvider: () => boolean;

  // Setters
  setApiKey: (provider: ProviderType, key: string) => void;
  setActiveProvider: (provider: ProviderType) => void;
  clearApiKey: (provider: ProviderType) => void;
  clearAllApiKeys: () => void;

  // State
  apiKeys: ApiKeysState;
  PROVIDERS: typeof PROVIDERS;
}

/**
 * Hook to manage API keys for OpenAI and Anthropic
 */
export function useApiKey(): UseApiKeyReturn {
  const [apiKeys, setApiKeys] = useState<ApiKeysState>(() => {
    try {
      const stored = localStorage.getItem(API_KEY_STORAGE_KEY);
      if (stored) {
        return JSON.parse(stored);
      }
    } catch (error) {
      console.error('Error loading API keys:', error);
    }
    return {
      openai: '',
      anthropic: '',
      gemini: '',
      activeProvider: null,
      lastUpdated: null
    };
  });

  // Save to localStorage whenever keys change
  useEffect(() => {
    try {
      localStorage.setItem(API_KEY_STORAGE_KEY, JSON.stringify(apiKeys));
    } catch (error) {
      console.error('Error saving API keys:', error);
    }
  }, [apiKeys]);

  /**
   * Set API key for a provider
   */
  const setApiKey = (provider: ProviderType, key: string): void => {
    setApiKeys(prev => ({
      ...prev,
      [provider]: key,
      // Auto-select this provider if none is active
      activeProvider: prev.activeProvider || (key ? provider : null),
      lastUpdated: new Date().toISOString()
    }));
  };

  /**
   * Get API key for a provider
   */
  const getApiKey = (provider: ProviderType): string => {
    return apiKeys[provider] || '';
  };

  /**
   * Check if a provider is configured
   */
  const isProviderConfigured = (provider: ProviderType): boolean => {
    return !!apiKeys[provider];
  };

  /**
   * Get the active provider
   */
  const getActiveProvider = (): ProviderType | null => {
    return apiKeys.activeProvider;
  };

  /**
   * Set the active provider
   */
  const setActiveProvider = (provider: ProviderType): void => {
    if (!isProviderConfigured(provider)) {
      console.warn(`Cannot set ${provider} as active - not configured`);
      return;
    }
    setApiKeys(prev => ({
      ...prev,
      activeProvider: provider
    }));
  };

  /**
   * Check if any provider is configured
   */
  const hasAnyProvider = (): boolean => {
    return isProviderConfigured(PROVIDERS.OPENAI) ||
           isProviderConfigured(PROVIDERS.ANTHROPIC) ||
           isProviderConfigured(PROVIDERS.GEMINI);
  };

  /**
   * Get list of configured providers
   */
  const getConfiguredProviders = (): ProviderType[] => {
    const providers: ProviderType[] = [];
    if (isProviderConfigured(PROVIDERS.OPENAI)) providers.push(PROVIDERS.OPENAI);
    if (isProviderConfigured(PROVIDERS.ANTHROPIC)) providers.push(PROVIDERS.ANTHROPIC);
    if (isProviderConfigured(PROVIDERS.GEMINI)) providers.push(PROVIDERS.GEMINI);
    return providers;
  };

  /**
   * Clear API key for a provider
   */
  const clearApiKey = (provider: ProviderType): void => {
    setApiKeys(prev => {
      const newKeys: ApiKeysState = { ...prev, [provider]: '' };

      // If we're clearing the active provider, switch to another one if available
      if (prev.activeProvider === provider) {
        const otherProviders = [PROVIDERS.OPENAI, PROVIDERS.ANTHROPIC, PROVIDERS.GEMINI]
          .filter(p => p !== provider && newKeys[p]);
        newKeys.activeProvider = otherProviders[0] || null;
      }

      return newKeys;
    });
  };

  /**
   * Clear all API keys
   */
  const clearAllApiKeys = (): void => {
    setApiKeys({
      openai: '',
      anthropic: '',
      gemini: '',
      activeProvider: null,
      lastUpdated: null
    });
  };

  return {
    // Getters
    getApiKey,
    getActiveProvider,
    getConfiguredProviders,
    isProviderConfigured,
    hasAnyProvider,

    // Setters
    setApiKey,
    setActiveProvider,
    clearApiKey,
    clearAllApiKeys,

    // State
    apiKeys,
    PROVIDERS
  };
}
