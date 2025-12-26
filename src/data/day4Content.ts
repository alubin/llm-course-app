import type { DayContent } from './types';

export const day4Content: DayContent = {
  id: 4,
  title: "AI-Powered REST API with Java Spring Boot",
  subtitle: "Build enterprise-grade AI APIs with Spring AI",
  duration: "6-8 hours",
  difficulty: "Intermediate",

  objectives: [
    "Learn Spring AI framework fundamentals",
    "Design production-ready AI API endpoints",
    "Implement rate limiting and caching",
    "Add comprehensive error handling",
    "Generate OpenAPI documentation",
    "Deploy with Docker and monitoring"
  ],

  prerequisites: [
    { name: "Java", details: "Java 17+ and basic Spring Boot knowledge" },
    { name: "REST APIs", details: "Understanding of HTTP and RESTful design" },
    { name: "Maven/Gradle", details: "Build tool basics" },
    { name: "Docker", details: "Basic container knowledge" }
  ],

  technologies: [
    { name: "Spring Boot 3.2", purpose: "Application framework" },
    { name: "Spring AI", purpose: "LLM integration for Spring" },
    { name: "OpenAI/Azure OpenAI", purpose: "LLM provider" },
    { name: "Redis", purpose: "Caching responses" },
    { name: "Micrometer", purpose: "Metrics and observability" },
    { name: "SpringDoc", purpose: "OpenAPI/Swagger docs" }
  ],

  sections: [
    {
      id: "theory",
      title: "Part 1: Theory — Enterprise AI APIs",
      estimatedTime: "1.5-2 hours",
      modules: [
        {
          id: "spring-ai-intro",
          title: "Introduction to Spring AI",
          content: `
### What is Spring AI?

Spring AI brings AI capabilities to the Spring ecosystem with:
- Unified API across LLM providers (OpenAI, Azure, Ollama, etc.)
- Spring-native patterns (dependency injection, auto-configuration)
- Production features (retry, caching, observability)

### Architecture

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                    SPRING AI ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   REST Controller                                               │
│        │                                                         │
│        ▼                                                         │
│   Service Layer  ──────► ChatClient (Spring AI)                 │
│        │                      │                                  │
│        │                      ▼                                  │
│        │                 ChatModel Interface                     │
│        │                      │                                  │
│        │                      ▼                                  │
│        │           ┌──────────┴──────────┐                       │
│        │           │                     │                       │
│        │      OpenAI Impl          Azure Impl                    │
│        │                                                         │
│        ▼                                                         │
│   Cache Layer (Redis)                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

### Key Components

\`\`\`java
// Auto-configured ChatClient
@Autowired
private ChatClient chatClient;

// Simple usage
String response = chatClient.call("What is Java?");

// Advanced usage with options
ChatResponse response = chatClient.call(
    new Prompt("Explain Spring Boot",
        ChatOptions.builder()
            .withModel("gpt-4")
            .withTemperature(0.7)
            .build()
    )
);
\`\`\`
          `
        },
        {
          id: "api-design",
          title: "API Design Patterns",
          content: `
### RESTful AI Endpoint Design

**1. Synchronous Chat**
\`\`\`
POST /api/v1/chat
Content-Type: application/json

{
  "message": "What is Java?",
  "model": "gpt-4",
  "temperature": 0.7
}

Response:
{
  "response": "Java is...",
  "model": "gpt-4",
  "usage": {
    "promptTokens": 10,
    "completionTokens": 50,
    "totalTokens": 60
  }
}
\`\`\`

**2. Streaming Chat**
\`\`\`
POST /api/v1/chat/stream
Content-Type: application/json

Response:
data: {"chunk": "Java"}
data: {"chunk": " is"}
data: {"chunk": " a"}
data: {"done": true}
\`\`\`

**3. Conversation Management**
\`\`\`
POST   /api/v1/conversations          # Create
GET    /api/v1/conversations/{id}     # Get
POST   /api/v1/conversations/{id}/messages  # Add message
DELETE /api/v1/conversations/{id}     # Delete
\`\`\`

### Error Response Format

\`\`\`json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "status": 429,
  "error": "Too Many Requests",
  "message": "Rate limit exceeded. Retry after 60 seconds.",
  "path": "/api/v1/chat"
}
\`\`\`

### Versioning Strategy

\`\`\`java
// URL versioning (recommended for breaking changes)
@RequestMapping("/api/v1")
public class ChatControllerV1 { }

@RequestMapping("/api/v2")
public class ChatControllerV2 { }
\`\`\`
          `
        },
        {
          id: "rate-limiting",
          title: "Rate Limiting & Caching",
          content: `
### Why Rate Limiting?

1. **Cost Control** - Prevent API abuse
2. **Fair Usage** - Ensure availability for all users
3. **Provider Limits** - Stay within OpenAI rate limits

### Rate Limiting Strategies

**Token Bucket Algorithm**

\`\`\`java
@Configuration
public class RateLimitConfig {

    @Bean
    public Bucket createBucket() {
        // 100 requests per minute
        Bandwidth limit = Bandwidth.classic(100,
            Refill.intervally(100, Duration.ofMinutes(1)));
        return Bucket.builder()
            .addLimit(limit)
            .build();
    }
}
\`\`\`

**Usage with Bucket4j**

\`\`\`java
@Service
public class RateLimitService {

    private final Bucket bucket;

    public boolean tryConsume() {
        return bucket.tryConsume(1);
    }

    public long getAvailableTokens() {
        return bucket.getAvailableTokens();
    }
}
\`\`\`

### Caching Strategies

**1. Response Caching**

\`\`\`java
@Cacheable(value = "chatResponses", key = "#message")
public String getCachedResponse(String message) {
    return chatClient.call(message);
}
\`\`\`

**2. Cache Configuration**

\`\`\`java
@Configuration
@EnableCaching
public class CacheConfig {

    @Bean
    public CacheManager cacheManager(RedisConnectionFactory factory) {
        RedisCacheConfiguration config = RedisCacheConfiguration
            .defaultCacheConfig()
            .entryTtl(Duration.ofHours(1))  // 1 hour TTL
            .serializeValuesWith(/* JSON serialization */);

        return RedisCacheManager.builder(factory)
            .cacheDefaults(config)
            .build();
    }
}
\`\`\`

**Cache Invalidation**

\`\`\`java
@CacheEvict(value = "chatResponses", allEntries = true)
public void clearCache() {
    // Cache cleared
}
\`\`\`
          `
        }
      ]
    },
    {
      id: "hands-on",
      title: "Part 2: Hands-On — Building the API",
      estimatedTime: "4.5-6 hours",
      tasks: [
        {
          id: "task-1",
          title: "Project Setup",
          description: "Initialize Spring Boot project",
          content: `
### Use Spring Initializr

Visit https://start.spring.io or use CLI:

\`\`\`bash
spring init \\
  --dependencies=web,data-jpa,validation,actuator,cache \\
  --group=com.example \\
  --artifact=ai-api \\
  --name=ai-api \\
  --package-name=com.example.aiapi \\
  ai-api
\`\`\`

### Add Spring AI Dependencies

Edit \`pom.xml\`:

\`\`\`xml
<dependencies>
    <!-- Spring AI -->
    <dependency>
        <groupId>org.springframework.ai</groupId>
        <artifactId>spring-ai-openai-spring-boot-starter</artifactId>
        <version>0.8.0</version>
    </dependency>

    <!-- Redis for caching -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-redis</artifactId>
    </dependency>

    <!-- Rate limiting -->
    <dependency>
        <groupId>com.github.vladimir-bukhtoyarov</groupId>
        <artifactId>bucket4j-core</artifactId>
        <version>8.7.0</version>
    </dependency>

    <!-- OpenAPI docs -->
    <dependency>
        <groupId>org.springdoc</groupId>
        <artifactId>springdoc-openapi-starter-webmvc-ui</artifactId>
        <version>2.3.0</version>
    </dependency>
</dependencies>
\`\`\`

### Application Properties

Create \`application.yml\`:

\`\`\`yaml
spring:
  application:
    name: ai-api
  ai:
    openai:
      api-key: \${OPENAI_API_KEY}
      chat:
        options:
          model: gpt-4
          temperature: 0.7
  redis:
    host: localhost
    port: 6379
  cache:
    type: redis

server:
  port: 8080

management:
  endpoints:
    web:
      exposure:
        include: health,metrics,prometheus
\`\`\`
          `
        },
        {
          id: "task-2",
          title: "DTOs and Models",
          description: "Create request/response objects",
          content: `
### Create ChatRequest DTO

\`\`\`java
package com.example.aiapi.dto;

import jakarta.validation.constraints.*;
import lombok.Data;

@Data
public class ChatRequest {

    @NotBlank(message = "Message cannot be blank")
    @Size(max = 4000, message = "Message too long")
    private String message;

    @Pattern(regexp = "gpt-3.5-turbo|gpt-4|gpt-4-turbo",
             message = "Invalid model")
    private String model = "gpt-4";

    @Min(0) @Max(2)
    private Double temperature = 0.7;

    @Min(1) @Max(4000)
    private Integer maxTokens = 1000;
}
\`\`\`

### Create ChatResponse DTO

\`\`\`java
package com.example.aiapi.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ChatResponse {

    private String response;
    private String model;
    private UsageInfo usage;
    private Long responseTimeMs;

    @Data
    @Builder
    public static class UsageInfo {
        private Integer promptTokens;
        private Integer completionTokens;
        private Integer totalTokens;
    }
}
\`\`\`

### Error Response DTO

\`\`\`java
package com.example.aiapi.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import java.time.Instant;

@Data
@Builder
@AllArgsConstructor
public class ErrorResponse {
    private Instant timestamp;
    private int status;
    private String error;
    private String message;
    private String path;
}
\`\`\`
          `
        },
        {
          id: "task-3",
          title: "Chat Service",
          description: "Implement core AI logic",
          content: `
### Create ChatService

\`\`\`java
package com.example.aiapi.service;

import com.example.aiapi.dto.ChatRequest;
import com.example.aiapi.dto.ChatResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.ChatClient;
import org.springframework.ai.chat.ChatResponse as AiChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;

@Slf4j
@Service
@RequiredArgsConstructor
public class ChatService {

    private final ChatClient chatClient;

    public ChatResponse chat(ChatRequest request) {
        long startTime = System.currentTimeMillis();

        // Build options
        OpenAiChatOptions options = OpenAiChatOptions.builder()
            .withModel(request.getModel())
            .withTemperature(request.getTemperature())
            .withMaxTokens(request.getMaxTokens())
            .build();

        // Call AI
        AiChatResponse aiResponse = chatClient.call(
            new Prompt(request.getMessage(), options)
        );

        long responseTime = System.currentTimeMillis() - startTime;

        // Build response
        return ChatResponse.builder()
            .response(aiResponse.getResult().getOutput().getContent())
            .model(request.getModel())
            .usage(ChatResponse.UsageInfo.builder()
                .promptTokens(aiResponse.getMetadata().getUsage().getPromptTokens())
                .completionTokens(aiResponse.getMetadata().getUsage().getGenerationTokens())
                .totalTokens(aiResponse.getMetadata().getUsage().getTotalTokens())
                .build())
            .responseTimeMs(responseTime)
            .build();
    }

    @Cacheable(value = "chatCache", key = "#message")
    public String chatCached(String message) {
        log.info("Cache miss for message: {}", message);
        return chatClient.call(message);
    }

    public Flux<String> chatStream(ChatRequest request) {
        OpenAiChatOptions options = OpenAiChatOptions.builder()
            .withModel(request.getModel())
            .withTemperature(request.getTemperature())
            .build();

        return chatClient.stream(new Prompt(request.getMessage(), options))
            .map(response -> response.getResult().getOutput().getContent());
    }
}
\`\`\`
          `
        },
        {
          id: "task-4",
          title: "REST Controller",
          description: "Build API endpoints",
          content: `
### Create ChatController

\`\`\`java
package com.example.aiapi.controller;

import com.example.aiapi.dto.ChatRequest;
import com.example.aiapi.dto.ChatResponse;
import com.example.aiapi.service.ChatService;
import com.example.aiapi.service.RateLimitService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

@Slf4j
@RestController
@RequestMapping("/api/v1")
@RequiredArgsConstructor
@Tag(name = "Chat", description = "AI Chat endpoints")
public class ChatController {

    private final ChatService chatService;
    private final RateLimitService rateLimitService;

    @PostMapping("/chat")
    @Operation(summary = "Send chat message", description = "Get AI response for a message")
    public ResponseEntity<ChatResponse> chat(@Valid @RequestBody ChatRequest request) {

        // Check rate limit
        if (!rateLimitService.tryConsume()) {
            return ResponseEntity.status(HttpStatus.TOO_MANY_REQUESTS).build();
        }

        log.info("Chat request: {}", request.getMessage());
        ChatResponse response = chatService.chat(request);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/chat/simple")
    @Operation(summary = "Simple chat", description = "Quick chat with default settings")
    public ResponseEntity<String> simpleChat(@RequestParam String message) {

        if (!rateLimitService.tryConsume()) {
            return ResponseEntity.status(HttpStatus.TOO_MANY_REQUESTS).build();
        }

        String response = chatService.chatCached(message);
        return ResponseEntity.ok(response);
    }

    @PostMapping(value = "/chat/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    @Operation(summary = "Stream chat", description = "Get streaming AI response")
    public Flux<String> streamChat(@Valid @RequestBody ChatRequest request) {

        if (!rateLimitService.tryConsume()) {
            return Flux.error(new RuntimeException("Rate limit exceeded"));
        }

        return chatService.chatStream(request);
    }

    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("OK");
    }
}
\`\`\`
          `
        },
        {
          id: "task-5",
          title: "Global Exception Handler",
          description: "Add comprehensive error handling",
          content: `
### Create GlobalExceptionHandler

\`\`\`java
package com.example.aiapi.exception;

import com.example.aiapi.dto.ErrorResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.context.request.WebRequest;

import java.time.Instant;
import java.util.stream.Collectors;

@Slf4j
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ErrorResponse> handleValidationErrors(
        MethodArgumentNotValidException ex,
        WebRequest request
    ) {
        String message = ex.getBindingResult().getFieldErrors().stream()
            .map(FieldError::getDefaultMessage)
            .collect(Collectors.joining(", "));

        ErrorResponse error = ErrorResponse.builder()
            .timestamp(Instant.now())
            .status(HttpStatus.BAD_REQUEST.value())
            .error("Validation Failed")
            .message(message)
            .path(request.getDescription(false))
            .build();

        return ResponseEntity.badRequest().body(error);
    }

    @ExceptionHandler(RateLimitExceededException.class)
    public ResponseEntity<ErrorResponse> handleRateLimit(
        RateLimitExceededException ex,
        WebRequest request
    ) {
        ErrorResponse error = ErrorResponse.builder()
            .timestamp(Instant.now())
            .status(HttpStatus.TOO_MANY_REQUESTS.value())
            .error("Too Many Requests")
            .message(ex.getMessage())
            .path(request.getDescription(false))
            .build();

        return ResponseEntity.status(HttpStatus.TOO_MANY_REQUESTS).body(error);
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleGenericException(
        Exception ex,
        WebRequest request
    ) {
        log.error("Unexpected error", ex);

        ErrorResponse error = ErrorResponse.builder()
            .timestamp(Instant.now())
            .status(HttpStatus.INTERNAL_SERVER_ERROR.value())
            .error("Internal Server Error")
            .message("An unexpected error occurred")
            .path(request.getDescription(false))
            .build();

        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(error);
    }
}
\`\`\`

### Custom Exceptions

\`\`\`java
package com.example.aiapi.exception;

public class RateLimitExceededException extends RuntimeException {
    public RateLimitExceededException(String message) {
        super(message);
    }
}
\`\`\`
          `
        },
        {
          id: "task-6",
          title: "Testing",
          description: "Write integration tests",
          content: `
### Create ChatControllerTest

\`\`\`java
package com.example.aiapi.controller;

import com.example.aiapi.dto.ChatRequest;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
class ChatControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void testChatEndpoint() throws Exception {
        ChatRequest request = new ChatRequest();
        request.setMessage("What is Java?");
        request.setModel("gpt-4");

        mockMvc.perform(post("/api/v1/chat")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.response").exists())
            .andExpect(jsonPath("$.usage").exists());
    }

    @Test
    void testValidation() throws Exception {
        ChatRequest request = new ChatRequest();
        request.setMessage("");  // Invalid: blank

        mockMvc.perform(post("/api/v1/chat")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
            .andExpect(status().isBadRequest());
    }
}
\`\`\`

### Run Tests

\`\`\`bash
mvn test
mvn verify
\`\`\`
          `
        },
        {
          id: "task-7",
          title: "Docker Deployment",
          description: "Containerize the application",
          content: `
### Create Dockerfile

\`\`\`dockerfile
FROM eclipse-temurin:17-jdk-alpine AS build
WORKDIR /app
COPY mvnw .
COPY .mvn .mvn
COPY pom.xml .
COPY src src
RUN ./mvnw package -DskipTests

FROM eclipse-temurin:17-jre-alpine
WORKDIR /app
COPY --from=build /app/target/*.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]
\`\`\`

### Create docker-compose.yml

\`\`\`yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - SPRING_REDIS_HOST=redis
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
\`\`\`

### Deploy

\`\`\`bash
docker-compose up -d
\`\`\`

### Access OpenAPI Docs

Visit http://localhost:8080/swagger-ui.html

🎉 **Congratulations!** Enterprise AI API complete!
          `
        }
      ]
    }
  ]
};
