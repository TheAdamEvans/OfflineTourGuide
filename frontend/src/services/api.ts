const API_URL = 'https://u47qxoiip6h5fk-8000.proxy.runpod.net/v1/chat/completions'

export interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export interface ChatCompletionChunk {
  id: string
  object: string
  created: number
  model: string
  choices: Array<{
    index: number
    delta: {
      role?: string
      content?: string
    }
    finish_reason: string | null
  }>
}

export async function* streamChatCompletion(
  messages: Message[],
  onError?: (error: Error) => void
): AsyncGenerator<string, void, unknown> {
  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        // model: 'Qwen/Qwen2.5-3B',
        model: 'runs/qwen3_pseudo3b',
        messages: messages,
        stream: true,
        temperature: 0.7,
        repetition_penalty: 1.15,
      }),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('No response body reader available')
    }

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      
      if (done) {
        break
      }

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        const trimmedLine = line.trim()
        if (trimmedLine === '' || !trimmedLine.startsWith('data: ')) {
          continue
        }

        if (trimmedLine === 'data: [DONE]') {
          return
        }

        try {
          const jsonStr = trimmedLine.slice(6) // Remove 'data: ' prefix
          const data: ChatCompletionChunk = JSON.parse(jsonStr)
          
          const content = data.choices[0]?.delta?.content
          if (content) {
            yield content
          }
        } catch (e) {
          // Skip invalid JSON lines
          console.warn('Failed to parse chunk:', trimmedLine, e)
        }
      }
    }
  } catch (error) {
    const err = error instanceof Error ? error : new Error('Unknown error')
    if (onError) {
      onError(err)
    } else {
      throw err
    }
  }
}

