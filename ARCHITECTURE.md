# LLM Course App - Architecture Documentation

## Overview

This is a React-based educational web application that teaches LLM engineering concepts through an interactive 7-day course with an AI-powered Python tutor. The app is built with TypeScript, React, and Vite, featuring a modern UI with animations and persistent progress tracking.

## Technology Stack

### Core Technologies
- **React 18.3.1** - UI framework with hooks and context API
- **TypeScript 5.6.2** - Type-safe JavaScript
- **Vite 5.4.2** - Fast build tool and dev server
- **Tailwind CSS 3.4.15** - Utility-first CSS framework
- **Framer Motion 11.11.17** - Animation library

### AI Provider SDKs
- **OpenAI SDK** (`openai@4.77.0`) - GPT-4 integration
- **Anthropic SDK** (`@anthropic-ai/sdk@0.32.1`) - Claude integration
- **Google Generative AI** (`@google/generative-ai@0.21.0`) - Gemini integration

### UI Components
- **Lucide React** - Icon library
- **React Syntax Highlighter** - Code syntax highlighting with Atom One Dark theme
- **React Router DOM 7.1.1** - Client-side routing

## Architecture Patterns

### Component Architecture

The application follows a hierarchical component structure:

```
App.tsx (Router)
├── Dashboard.tsx (Home page)
├── Roadmap.tsx (Course overview)
├── CoursePage.tsx (Day content viewer)
└── PythonPrimer.tsx (AI-powered Python tutorial)
    ├── ApiKeyModal.tsx (API key configuration)
    ├── TutorResults.tsx (AI response display)
    ├── SearchHistory.tsx (Previous questions)
    └── ProviderBadge.tsx (AI provider indicator)
```

### State Management

#### Local Storage Pattern
The app uses browser `localStorage` for persistence:

1. **API Keys** (`llm-course-api-keys`)
   - Stores OpenAI, Anthropic, and Gemini API keys
   - Managed by `useApiKey` hook
   - Never sent to external servers (client-side only)
   - Structure:
   ```typescript
   {
     openai: string,
     anthropic: string,
     gemini: string,
     activeProvider: 'openai' | 'anthropic' | 'gemini' | null,
     lastUpdated: string | null
   }
   ```

2. **Progress Tracking** (`llm-course-progress`)
   - Tracks completed tasks and days
   - Managed by `useProgress` hook
   - Structure:
   ```typescript
   {
     completedTasks: Record<number, string[]>,  // dayId -> taskIds[]
     completedDays: number[],
     currentDay: number,
     lastVisited: string | null,
     totalTimeSpent: number
   }
   ```

3. **Search History** (`llm-tutor-history`)
   - Stores AI tutor Q&A history
   - Managed by `SearchHistoryContext`
   - Structure:
   ```typescript
   {
     question: string,
     answer: string,
     timestamp: string,
     provider: 'openai' | 'anthropic' | 'gemini'
   }[]
   ```

#### React Context API
Used for sharing state across component trees:

- **ProgressContext** - Course progress state
- **SearchHistoryContext** - AI tutor conversation history

### AI Provider Abstraction

The `pythonTutor.ts` service implements a clean abstraction over three AI providers:

```typescript
export type AIProvider = 'openai' | 'anthropic' | 'gemini';

export interface TutorResponse {
  answer: string;
  provider: AIProvider;
}

class PythonTutorService {
  async askQuestion(question: string, provider: AIProvider, apiKey: string): Promise<TutorResponse>
  private async askWithOpenAI(question: string, apiKey: string): Promise<TutorResponse>
  private async askWithAnthropic(question: string, apiKey: string): Promise<TutorResponse>
  private async askWithGemini(question: string, apiKey: string): Promise<TutorResponse>
}
```

**Design Decisions:**
- Each provider uses its native SDK (not HTTP fetch)
- Consistent prompt engineering across providers
- Provider-agnostic response format
- Streaming responses where supported (OpenAI, Anthropic)
- Standardized error handling

**Provider Configurations:**
- **OpenAI**: GPT-4, temperature 0.7, max_tokens 2000
- **Anthropic**: Claude 3.5 Sonnet, temperature 0.7, max_tokens 2000
- **Gemini**: Gemini 2.0 Flash Exp, temperature 0.7, top_p 0.95

### TypeScript Migration Strategy

The codebase underwent a **full TypeScript migration** with these key decisions:

1. **JSX Configuration**: Uses `"jsx": "react-jsx"` (React 17+ JSX transform)
   - No need to import React in every file
   - Return types use `React.ReactElement` instead of `JSX.Element`

2. **Type Safety Approach**:
   - Strict mode enabled in `tsconfig.json`
   - Centralized types in `src/data/types.ts`
   - Explicit interfaces for all props and state
   - No `any` types allowed

3. **Gradual Migration Path**:
   - Data layer first (types, roadmap, content)
   - Hooks second (custom hooks with typed returns)
   - Components last (UI layer)
   - Removed `.jsx` files only after TypeScript compilation passed

4. **Key Type Definitions**:
   ```typescript
   // Core content types
   export interface CourseDay { ... }
   export interface DayContent { ... }
   export interface Section { ... }
   export interface Module { ... }
   export interface Task { ... }

   // Hook return types
   export interface UseProgressReturn { ... }
   export interface UseApiKeyReturn { ... }
   ```

## Data Layer

### Content Structure

Course content is organized in separate TypeScript modules:

```
src/data/
├── types.ts                 # Shared type definitions
├── courseRoadmap.ts         # Course overview data
├── day0Content.ts           # Day 0: Why Python for AI?
├── day1Content.ts           # Day 1: CLI Assistant
├── day2Content.ts           # Day 2: Chatbot with Memory
├── day3Content.ts           # Day 3: RAG System
├── day4Content.ts           # Day 4: REST API
├── day5Content.ts           # Day 5: Transformers & Fine-tuning
├── day6Content.ts           # Day 6: AI Agents
├── day7Content.ts           # Day 7: Data Pipeline
└── pythonPrimer.ts          # Python tutorial topics
```

**Content Format:**
- Markdown-based content with inline formatting
- Code examples with syntax highlighting
- Structured sections and modules
- Tasks with checkboxes for progress tracking

### Inline Formatting Parser

Custom markdown parser for rich text rendering:

```typescript
// Supports: **bold**, *italic*, `code`, [links](url)
function parseInlineFormatting(text: string): React.ReactNode[]
```

Used in `CoursePage.tsx` and `PythonPrimer.tsx` for rendering content.

## Custom Hooks

### `useProgress()`
**Purpose**: Manage course progress tracking

**Returns**:
- `progress` - Current progress state
- `toggleTask()` - Mark task as complete/incomplete
- `isTaskCompleted()` - Check task status
- `getCompletedTasksCount()` - Count completed tasks for a day
- `markDayComplete()` - Mark entire day complete
- `isDayComplete()` - Check day status
- `setCurrentDay()` - Update current day
- `resetProgress()` - Clear all progress
- `getOverallProgress()` - Calculate completion percentage

**Storage**: Persists to `localStorage` on every change via `useEffect`

### `useApiKey()`
**Purpose**: Manage AI provider API keys

**Returns**:
- `apiKeys` - Current API keys state
- `setApiKey()` - Store API key for a provider
- `clearApiKey()` - Remove API key
- `getActiveProvider()` - Get selected provider
- `setActiveProvider()` - Change provider
- `hasAnyProvider()` - Check if any key exists
- `hasProvider()` - Check if specific provider has key

**Security**: Keys stored in browser `localStorage` only, never transmitted to servers

## Routing

Client-side routing with React Router:

```typescript
<BrowserRouter>
  <Routes>
    <Route path="/" element={<Dashboard />} />
    <Route path="/roadmap" element={<Roadmap />} />
    <Route path="/course/day/:dayId" element={<CoursePage />} />
    <Route path="/python-primer" element={<PythonPrimer />} />
    <Route path="/python-primer/:topicId" element={<PythonPrimer />} />
  </Routes>
</BrowserRouter>
```

**Navigation Pattern**:
- Dashboard → Roadmap → Course Day → Lessons
- Python Primer accessible from sidebar navigation
- Deep linking supported for all routes

## UI/UX Design Patterns

### Animation Strategy (Framer Motion)

1. **Page Transitions**:
   ```typescript
   <motion.div
     initial={{ opacity: 0, y: 20 }}
     animate={{ opacity: 1, y: 0 }}
     transition={{ duration: 0.5 }}
   >
   ```

2. **Staggered Lists**:
   ```typescript
   <motion.div
     initial={{ opacity: 0 }}
     animate={{ opacity: 1 }}
     transition={{ delay: index * 0.1 }}
   >
   ```

3. **Interactive Elements**:
   - Hover effects on cards
   - Scale animations on buttons
   - Smooth state transitions

### Color Scheme & Theming

- **Background**: Dark theme (`bg-gray-950`)
- **Accent Colors**:
  - OpenAI: Green (`text-green-400`)
  - Anthropic: Orange (`text-orange-400`)
  - Gemini: Blue (`text-blue-400`)
- **UI Elements**: Gray scale with cyan/purple accents
- **Code Blocks**: Atom One Dark theme

### Responsive Design

- **Desktop-first** with mobile breakpoints
- Tailwind responsive utilities (`sm:`, `md:`, `lg:`)
- Flexible grid layouts for course cards
- Mobile-friendly navigation

## Build & Development

### Development Workflow

```bash
npm install          # Install dependencies
npm run dev          # Start Vite dev server (http://localhost:5173)
npm run build        # Production build (dist/)
npm run preview      # Preview production build
npx tsc --noEmit     # Type check without emitting files
```

### Build Configuration

**Vite Config** (`vite.config.ts`):
```typescript
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    open: true
  }
})
```

**TypeScript Config** (`tsconfig.json`):
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "allowJs": true
  }
}
```

### Environment Variables

**Not used** - API keys are user-provided and stored in browser localStorage.

**Rationale**:
- Keeps the app purely client-side (no backend needed)
- Users bring their own API keys (BYOK model)
- No server costs or API rate limits to manage

## Design Decisions & Rationale

### Why Client-Side Only?

**Decision**: No backend server, all computation in browser

**Rationale**:
1. **Simplicity**: Static hosting (Vercel, Netlify, GitHub Pages)
2. **Privacy**: User API keys never leave their browser
3. **Cost**: No server infrastructure needed
4. **Performance**: Direct API calls to AI providers
5. **Scalability**: Unlimited users without server costs

### Why Multiple AI Providers?

**Decision**: Support OpenAI, Anthropic, and Gemini

**Rationale**:
1. **Flexibility**: Users can choose based on preference/cost
2. **Reliability**: Fallback if one provider has issues
3. **Comparison**: Learn differences between providers
4. **Educational**: Understanding multiple LLM APIs

### Why Task IDs are Strings?

**Decision**: Changed from `number` to `string` during TypeScript migration

**Rationale**:
1. Content files use string IDs for flexibility ("task-1", "intro", etc.)
2. Avoids type coercion issues
3. More semantic (can use descriptive IDs)
4. Consistent with section/module IDs

### Why Separate Content Files?

**Decision**: Each day has its own TypeScript file

**Rationale**:
1. **Maintainability**: Easy to find and edit specific day content
2. **Code Splitting**: Vite can lazy-load day content
3. **Collaboration**: Multiple people can work on different days
4. **Type Safety**: Each file exports typed `DayContent`

## Performance Considerations

### Bundle Size Optimization

1. **Code Splitting**: React Router enables route-based splitting
2. **Tree Shaking**: Vite automatically removes unused code
3. **Lazy Loading**: Content files loaded on-demand
4. **Icon Optimization**: Lucide imports only used icons

### Runtime Performance

1. **LocalStorage Batching**: Updates batched via `useEffect`
2. **Memoization**: Could add `useMemo`/`useCallback` for expensive operations
3. **Virtual Scrolling**: Not needed (content pages are reasonably sized)

### AI Response Streaming

- **OpenAI & Anthropic**: Use streaming APIs for better UX
- **Gemini**: Uses `streamGenerateContent` for progressive rendering
- **User Feedback**: Shows partial responses as they arrive

## Security Considerations

### API Key Storage

**Current Approach**: Browser `localStorage`

**Security Implications**:
- ✅ Not stored on any server
- ✅ Only accessible to same-origin JavaScript
- ⚠️ Vulnerable to XSS attacks
- ⚠️ Not encrypted at rest
- ⚠️ Cleared if user clears browser data

**Mitigation**:
- No third-party scripts that could exfiltrate keys
- Content Security Policy (CSP) headers recommended for production
- User education: API keys are their responsibility

### XSS Prevention

1. React automatically escapes JSX expressions
2. No `dangerouslySetInnerHTML` usage
3. Content is statically defined (not user-generated)
4. No `eval()` or dynamic code execution

## Future Enhancements

### Potential Improvements

1. **Backend Option**: Optional backend for users without API keys
2. **Progress Sync**: Cloud sync across devices
3. **Code Execution**: In-browser Python REPL (Pyodide)
4. **Video Content**: Embedded tutorial videos
5. **Exercises**: Interactive coding challenges
6. **Certifications**: Downloadable completion certificates
7. **Community**: Share progress, leaderboards
8. **Offline Mode**: Service worker for offline access

### Technical Debt

1. **Testing**: Add unit tests (Vitest), integration tests (Playwright)
2. **Error Boundaries**: React error boundaries for graceful failures
3. **Analytics**: Optional usage tracking (privacy-respecting)
4. **A11y**: Improve keyboard navigation and screen reader support
5. **i18n**: Internationalization for multiple languages

## Troubleshooting

### Common Issues

1. **API Key Modal Keeps Appearing**
   - **Cause**: `apiKeyHook` in useEffect dependencies
   - **Fix**: Remove from dependency array in `PythonPrimer.tsx:212`

2. **TypeScript Compilation Errors**
   - **Cause**: Missing type definitions
   - **Fix**: Install `@types/*` packages or add to `tsconfig.json`

3. **LocalStorage Not Persisting**
   - **Cause**: Browser privacy mode or storage quota exceeded
   - **Fix**: Check browser settings, clear old data

4. **AI Responses Failing**
   - **Cause**: Invalid API key or rate limits
   - **Fix**: Verify API key, check provider dashboard

## Contributing Guidelines

1. **TypeScript**: All new code must be TypeScript
2. **Types**: Define interfaces for all props and state
3. **Formatting**: Use Prettier (recommended)
4. **Linting**: Run `npx tsc --noEmit` before committing
5. **Content**: Follow existing markdown formatting patterns
6. **Commits**: Use conventional commits format

---

**Last Updated**: 2025-12-26
**Version**: 1.0.0
**Maintained By**: Andy Lubin
