import { useState, useEffect } from 'react';

const API_KEY_STORAGE_KEY = 'llm_course_api_keys';

export const PROVIDERS = {
  OPENAI: 'openai',
  ANTHROPIC: 'anthropic'
};

/**
 * Hook to manage API keys for OpenAI and Anthropic
 */
export function useApiKey() {
  const [apiKeys, setApiKeys] = useState(() => {
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
  const setApiKey = (provider, key) => {
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
  const getApiKey = (provider) => {
    return apiKeys[provider] || '';
  };

  /**
   * Check if a provider is configured
   */
  const isProviderConfigured = (provider) => {
    return !!apiKeys[provider];
  };

  /**
   * Get the active provider
   */
  const getActiveProvider = () => {
    return apiKeys.activeProvider;
  };

  /**
   * Set the active provider
   */
  const setActiveProvider = (provider) => {
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
  const hasAnyProvider = () => {
    return isProviderConfigured(PROVIDERS.OPENAI) || isProviderConfigured(PROVIDERS.ANTHROPIC);
  };

  /**
   * Get list of configured providers
   */
  const getConfiguredProviders = () => {
    const providers = [];
    if (isProviderConfigured(PROVIDERS.OPENAI)) providers.push(PROVIDERS.OPENAI);
    if (isProviderConfigured(PROVIDERS.ANTHROPIC)) providers.push(PROVIDERS.ANTHROPIC);
    return providers;
  };

  /**
   * Clear API key for a provider
   */
  const clearApiKey = (provider) => {
    setApiKeys(prev => {
      const newKeys = { ...prev, [provider]: '' };

      // If we're clearing the active provider, switch to the other one if available
      if (prev.activeProvider === provider) {
        const otherProvider = provider === PROVIDERS.OPENAI ? PROVIDERS.ANTHROPIC : PROVIDERS.OPENAI;
        newKeys.activeProvider = newKeys[otherProvider] ? otherProvider : null;
      }

      return newKeys;
    });
  };

  /**
   * Clear all API keys
   */
  const clearAllApiKeys = () => {
    setApiKeys({
      openai: '',
      anthropic: '',
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
