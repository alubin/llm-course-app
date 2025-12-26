import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';

const SYSTEM_PROMPT = `You are a patient Python tutor for LLM engineering students.
Explain Python concepts clearly and concisely with:
1. Simple explanation in 2-3 sentences
2. Practical code example (runnable and copy-pastable)
3. Common pitfalls to avoid
4. How this relates to LLM/AI development (if applicable)

Always provide working code examples. Keep explanations concise but complete.

Respond in JSON format:
{
  "explanation": "Clear 2-3 sentence explanation",
  "code": "# Runnable Python code example\\nprint('example')",
  "pitfalls": "Common mistakes to avoid",
  "llm_connection": "How this relates to LLM development (or null if not applicable)",
  "related_topics": ["topic1", "topic2"]
}`;

/**
 * Python Tutor service that supports both OpenAI and Anthropic
 */
class PythonTutorService {
  /**
   * Ask a Python question using OpenAI
   */
  async askWithOpenAI(question, apiKey) {
    if (!apiKey) {
      throw new Error('OpenAI API key is required');
    }

    const client = new OpenAI({
      apiKey,
      dangerouslyAllowBrowser: true // Client-side usage
    });

    try {
      const response = await client.chat.completions.create({
        model: 'gpt-4-turbo-preview',
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          { role: 'user', content: question }
        ],
        response_format: { type: 'json_object' },
        temperature: 0.7,
        max_tokens: 1500
      });

      const content = response.choices[0].message.content;
      const parsed = JSON.parse(content);

      return {
        provider: 'openai',
        model: 'gpt-4-turbo-preview',
        ...parsed,
        tokens: response.usage.total_tokens,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error('OpenAI error:', error);
      throw new Error(`OpenAI request failed: ${error.message}`);
    }
  }

  /**
   * Ask a Python question using Anthropic Claude
   */
  async askWithAnthropic(question, apiKey) {
    if (!apiKey) {
      throw new Error('Anthropic API key is required');
    }

    const client = new Anthropic({
      apiKey,
      dangerouslyAllowBrowser: true // Client-side usage
    });

    try {
      const response = await client.messages.create({
        model: 'claude-3-sonnet-20240229',
        max_tokens: 1500,
        temperature: 0.7,
        system: SYSTEM_PROMPT,
        messages: [
          { role: 'user', content: question }
        ]
      });

      const content = response.content[0].text;

      // Try to parse as JSON
      let parsed;
      try {
        parsed = JSON.parse(content);
      } catch (e) {
        // If not valid JSON, wrap the response
        parsed = {
          explanation: content,
          code: null,
          pitfalls: null,
          llm_connection: null,
          related_topics: []
        };
      }

      return {
        provider: 'anthropic',
        model: 'claude-3-sonnet-20240229',
        ...parsed,
        tokens: response.usage.input_tokens + response.usage.output_tokens,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error('Anthropic error:', error);
      throw new Error(`Anthropic request failed: ${error.message}`);
    }
  }

  /**
   * Ask a Python question using the specified provider
   */
  async ask(question, provider, apiKey) {
    if (!question || !question.trim()) {
      throw new Error('Question cannot be empty');
    }

    if (provider === 'openai') {
      return this.askWithOpenAI(question, apiKey);
    } else if (provider === 'anthropic') {
      return this.askWithAnthropic(question, apiKey);
    } else {
      throw new Error(`Unknown provider: ${provider}`);
    }
  }
}

// Export a singleton instance
export const pythonTutor = new PythonTutorService();
