import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ExternalLink, Eye, EyeOff, Check, Sparkles, Zap } from 'lucide-react';

export default function ApiKeyModal({ isOpen, onClose, useApiKey }) {
  const [activeTab, setActiveTab] = useState('openai');
  const [openaiKey, setOpenaiKey] = useState(useApiKey.getApiKey('openai'));
  const [anthropicKey, setAnthropicKey] = useState(useApiKey.getApiKey('anthropic'));
  const [showOpenAIKey, setShowOpenAIKey] = useState(false);
  const [showAnthropicKey, setShowAnthropicKey] = useState(false);

  const handleSave = () => {
    if (openaiKey) {
      useApiKey.setApiKey('openai', openaiKey);
    }
    if (anthropicKey) {
      useApiKey.setApiKey('anthropic', anthropicKey);
    }
    onClose();
  };

  const handleSkip = () => {
    onClose();
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
          >
            <div className="bg-surface-900 border border-surface-700 rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-auto">
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-surface-800">
                <div>
                  <h2 className="text-2xl font-display font-bold text-white">
                    AI Tutor Setup
                  </h2>
                  <p className="text-surface-400 mt-1">
                    Configure your AI provider to enable the Python tutor
                  </p>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 hover:bg-surface-800 rounded-lg transition-colors"
                >
                  <X size={20} className="text-surface-400" />
                </button>
              </div>

              {/* Tabs */}
              <div className="flex border-b border-surface-800">
                <button
                  onClick={() => setActiveTab('openai')}
                  className={`flex-1 px-6 py-4 font-medium transition-colors relative ${
                    activeTab === 'openai'
                      ? 'text-emerald-400'
                      : 'text-surface-400 hover:text-white'
                  }`}
                >
                  <span className="flex items-center justify-center gap-2">
                    <Sparkles size={18} />
                    OpenAI (GPT-4)
                  </span>
                  {activeTab === 'openai' && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute bottom-0 left-0 right-0 h-0.5 bg-emerald-400"
                    />
                  )}
                </button>
                <button
                  onClick={() => setActiveTab('anthropic')}
                  className={`flex-1 px-6 py-4 font-medium transition-colors relative ${
                    activeTab === 'anthropic'
                      ? 'text-orange-400'
                      : 'text-surface-400 hover:text-white'
                  }`}
                >
                  <span className="flex items-center justify-center gap-2">
                    <Zap size={18} />
                    Anthropic (Claude)
                  </span>
                  {activeTab === 'anthropic' && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute bottom-0 left-0 right-0 h-0.5 bg-orange-400"
                    />
                  )}
                </button>
              </div>

              {/* Content */}
              <div className="p-6 space-y-6">
                {activeTab === 'openai' ? (
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-surface-300 mb-2">
                        OpenAI API Key
                      </label>
                      <div className="relative">
                        <input
                          type={showOpenAIKey ? 'text' : 'password'}
                          value={openaiKey}
                          onChange={(e) => setOpenaiKey(e.target.value)}
                          placeholder="sk-..."
                          className="w-full px-4 py-3 bg-surface-800 border border-surface-700 rounded-lg text-white placeholder-surface-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 pr-10"
                        />
                        <button
                          onClick={() => setShowOpenAIKey(!showOpenAIKey)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-surface-400 hover:text-white"
                        >
                          {showOpenAIKey ? <EyeOff size={18} /> : <Eye size={18} />}
                        </button>
                      </div>
                    </div>

                    <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-4">
                      <h4 className="text-emerald-400 font-medium mb-2">How to get your API key:</h4>
                      <ol className="text-sm text-surface-300 space-y-1 list-decimal list-inside">
                        <li>Go to <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" className="text-emerald-400 hover:underline inline-flex items-center gap-1">platform.openai.com/api-keys <ExternalLink size={12} /></a></li>
                        <li>Sign in or create an account</li>
                        <li>Click "Create new secret key"</li>
                        <li>Copy and paste it above</li>
                      </ol>
                      <p className="text-xs text-surface-400 mt-3">
                        ✨ Your API key is stored locally in your browser and never sent to our servers.
                      </p>
                    </div>

                    {useApiKey.isProviderConfigured('openai') && (
                      <div className="flex items-center gap-2 text-emerald-400 text-sm">
                        <Check size={16} />
                        <span>OpenAI configured</span>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-surface-300 mb-2">
                        Anthropic API Key
                      </label>
                      <div className="relative">
                        <input
                          type={showAnthropicKey ? 'text' : 'password'}
                          value={anthropicKey}
                          onChange={(e) => setAnthropicKey(e.target.value)}
                          placeholder="sk-ant-..."
                          className="w-full px-4 py-3 bg-surface-800 border border-surface-700 rounded-lg text-white placeholder-surface-500 focus:outline-none focus:ring-2 focus:ring-orange-500 pr-10"
                        />
                        <button
                          onClick={() => setShowAnthropicKey(!showAnthropicKey)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-surface-400 hover:text-white"
                        >
                          {showAnthropicKey ? <EyeOff size={18} /> : <Eye size={18} />}
                        </button>
                      </div>
                    </div>

                    <div className="bg-orange-500/10 border border-orange-500/20 rounded-lg p-4">
                      <h4 className="text-orange-400 font-medium mb-2">How to get your API key:</h4>
                      <ol className="text-sm text-surface-300 space-y-1 list-decimal list-inside">
                        <li>Go to <a href="https://console.anthropic.com/settings/keys" target="_blank" rel="noopener noreferrer" className="text-orange-400 hover:underline inline-flex items-center gap-1">console.anthropic.com/settings/keys <ExternalLink size={12} /></a></li>
                        <li>Sign in or create an account</li>
                        <li>Click "Create Key"</li>
                        <li>Copy and paste it above</li>
                      </ol>
                      <p className="text-xs text-surface-400 mt-3">
                        ✨ Your API key is stored locally in your browser and never sent to our servers.
                      </p>
                    </div>

                    {useApiKey.isProviderConfigured('anthropic') && (
                      <div className="flex items-center gap-2 text-orange-400 text-sm">
                        <Check size={16} />
                        <span>Anthropic configured</span>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Footer */}
              <div className="flex items-center justify-between p-6 border-t border-surface-800">
                <button
                  onClick={handleSkip}
                  className="px-4 py-2 text-surface-400 hover:text-white transition-colors"
                >
                  Skip for now
                </button>
                <div className="flex gap-3">
                  <button
                    onClick={onClose}
                    className="px-4 py-2 bg-surface-800 hover:bg-surface-700 text-white rounded-lg transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleSave}
                    className="px-6 py-2 bg-brand-500 hover:bg-brand-600 text-white font-medium rounded-lg transition-colors"
                  >
                    Save
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
