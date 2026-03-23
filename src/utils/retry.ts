import { APICallError } from "@ai-sdk/provider"

export interface RetryOptions {
  maxRetries: number
  shouldRetry: (error: unknown) => boolean
  getDelayMs?: (attempt: number, error: unknown) => number
  abortSignal?: AbortSignal
}

export async function withRetries<T>(
  operation: () => Promise<T>,
  options: RetryOptions,
): Promise<T> {
  const { maxRetries, shouldRetry, getDelayMs, abortSignal } = options

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await operation()
    }
    catch (error) {
      if (abortSignal?.aborted) {
        throw abortSignal.reason ?? error
      }
      if (attempt === maxRetries || !shouldRetry(error)) {
        throw error
      }

      const delayMs = getDelayMs?.(attempt, error) ?? defaultRetryDelayMs(attempt)
      if (delayMs > 0) {
        await sleep(delayMs)
      }
    }
  }

  throw new Error("Retry loop exhausted.")
}

export function isRetryableQwenRequestError(error: unknown): boolean {
  if (APICallError.isInstance(error)) {
    const statusCode = error.statusCode
    if (statusCode === 408 || statusCode === 429) {
      return true
    }
    if (statusCode != null && statusCode >= 500) {
      return true
    }
    return false
  }

  if (error instanceof TypeError) {
    return true
  }

  return false
}

export function isRetryableQwenStreamBoundaryFailure(error: unknown): boolean {
  if (!APICallError.isInstance(error)) {
    return false
  }
  if (error.statusCode !== 500) {
    return false
  }

  const message = String(error.message ?? "")
  return (
    message.includes("list index out of range")
    && (message.includes("InternalServerError") || message.toLowerCase().includes("internal server error"))
  )
}

function defaultRetryDelayMs(attempt: number): number {
  return 100 * (attempt + 1)
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}
