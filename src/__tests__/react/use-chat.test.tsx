import type { FetchFunction } from "@ai-sdk/provider-utils"
import type { ChatTransport, UIMessage } from "ai"
import { useChat } from "@ai-sdk/react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { convertToModelMessages, streamText } from "ai"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { createQwen } from "../../provider"

describe("@ai-sdk/react", () => {
  beforeEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  function createQwenChatTransport(fetchImpl: FetchFunction): ChatTransport<UIMessage> {
    const provider = createQwen({
      apiKey: "test-api-key",
      baseURL: "https://my.api.com/v1/",
      fetch: fetchImpl,
    })
    const model = provider("qwen-chat")

    return {
      sendMessages: async ({ messages, abortSignal }) => {
        const modelMessages = await convertToModelMessages(
          messages.map(({ id: _id, ...message }) => message),
        )

        const result = streamText({
          model,
          messages: modelMessages,
          abortSignal,
          maxRetries: 0,
        })

        return result.toUIMessageStream()
      },
      reconnectToStream: async () => null,
    }
  }

  it("useChat exposes stable initial state", () => {
    const transport: ChatTransport<UIMessage> = {
      sendMessages: async () => new ReadableStream(),
      reconnectToStream: async () => null,
    }
    const { result } = renderHook(() => useChat({ transport }))

    expect(typeof result.current.id).toBe("string")
    expect(Array.isArray(result.current.messages)).toBe(true)
    expect(result.current.messages).toHaveLength(0)
    expect(result.current.status).toBe("ready")
    expect(result.current.error).toBeUndefined()
    expect(typeof result.current.sendMessage).toBe("function")
    expect(typeof result.current.regenerate).toBe("function")
    expect(typeof result.current.stop).toBe("function")
    expect(typeof result.current.clearError).toBe("function")
  })

  it("useChat allows setting messages locally", () => {
    const transport: ChatTransport<UIMessage> = {
      sendMessages: async () => new ReadableStream(),
      reconnectToStream: async () => null,
    }
    const { result } = renderHook(() => useChat({ transport }))

    act(() => {
      result.current.setMessages([
        { id: "m1", role: "user", parts: [{ type: "text", text: "hi" }] },
      ])
    })

    expect(result.current.messages).toHaveLength(1)
    expect(result.current.messages[0]?.role).toBe("user")
    const firstPart = result.current.messages[0]?.parts[0]
    expect(firstPart?.type === "text" ? firstPart.text : undefined).toBe("hi")
  })

  it("useChat streams assistant text using qwen chat model", async () => {
    let requestBody: any
    const responseChunks = [
      `data: {"id":"abc","object":"chat.completion.chunk","created":1702657020,"model":"qwen-chat","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n`,
      `data: {"id":"abc","object":"chat.completion.chunk","created":1702657021,"model":"qwen-chat","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n`,
      "data: [DONE]\n\n",
    ]

    const transport = createQwenChatTransport(async (_url: RequestInfo | URL, init?: RequestInit) => {
      requestBody = init?.body ? JSON.parse(init.body as string) : undefined

      const encoder = new TextEncoder()
      const stream = new ReadableStream({
        start(controller) {
          for (const chunk of responseChunks) controller.enqueue(encoder.encode(chunk))
          controller.close()
        },
      })

      return new Response(stream, {
        headers: { "content-type": "text/event-stream" },
      })
    })

    const { result } = renderHook(() => useChat({ transport }))

    await act(async () => {
      await result.current.sendMessage({ text: "hello" })
    })

    await waitFor(() => {
      expect(result.current.status).toBe("ready")
      expect(result.current.error).toBeUndefined()
    })

    expect(requestBody).toMatchObject({
      stream: true,
      stream_options: { include_usage: true },
    })

    const messages = result.current.messages
    let assistant: UIMessage | undefined
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i]
      if (m?.role === "assistant") {
        assistant = m
        break
      }
    }
    let assistantText = ""
    if (assistant) {
      for (const part of assistant.parts) {
        if (part.type === "text") {
          assistantText += part.text
        }
      }
    }

    expect(assistantText).toBe("ok")
  })

  it("useChat retries retryable qwen errors inside the provider", async () => {
    let fetchCalls = 0
    const responseChunks = [
      `data: {"id":"abc","object":"chat.completion.chunk","created":1702657020,"model":"qwen-chat","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n`,
      `data: {"id":"abc","object":"chat.completion.chunk","created":1702657021,"model":"qwen-chat","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n`,
      "data: [DONE]\n\n",
    ]

    const transport = createQwenChatTransport(async () => {
      fetchCalls++

      if (fetchCalls === 1) {
        return new Response(JSON.stringify({
          object: "error",
          message: "InternalServerError: list index out of range",
          type: "InternalServerError",
          param: null,
          code: null,
        }), {
          status: 500,
          headers: { "content-type": "application/json" },
        })
      }

      const encoder = new TextEncoder()
      const stream = new ReadableStream({
        start(controller) {
          for (const chunk of responseChunks) controller.enqueue(encoder.encode(chunk))
          controller.close()
        },
      })

      return new Response(stream, {
        headers: { "content-type": "text/event-stream" },
      })
    })

    const { result } = renderHook(() => useChat({ transport }))

    await act(async () => {
      await result.current.sendMessage({ text: "hello" })
    })

    await waitFor(() => {
      expect(result.current.status).toBe("ready")
      expect(result.current.error).toBeUndefined()
    })

    expect(fetchCalls).toBe(2)
  })
})
