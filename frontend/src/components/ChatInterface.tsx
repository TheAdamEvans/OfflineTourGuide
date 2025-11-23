import { useState, useRef, useEffect } from 'react'
import { streamChatCompletion, type Message } from '../services/api'
import { coordinatesToPlusCode, getPlaceName } from '../utils/location'
import './ChatInterface.css'

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  isStreaming?: boolean
  isHidden?: boolean
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showInput, setShowInput] = useState(false)
  const [showGoButton, setShowGoButton] = useState(true)
  const [hasStarted, setHasStarted] = useState(false)
  const [textToSpeechEnabled, setTextToSpeechEnabled] = useState(false)
  const [currentLocation, setCurrentLocation] = useState<string | null>(null)
  const [pullDistance, setPullDistance] = useState(0)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const hasInitialized = useRef(false)
  const speechSynthesisRef = useRef<SpeechSynthesisUtterance | null>(null)
  const chatMessagesRef = useRef<HTMLDivElement>(null)
  const touchStartY = useRef<number>(0)
  const touchStartScrollTop = useRef<number>(0)
  const isPulling = useRef(false)

  const TOUR_GUIDE_INSTRUCTIONS = `
  As a local tour guide, briefly describe this location. \
  Focus on things like the most interesting facts, a vivid sensory detail, \
  a personal or historical note, \
  and/or a practical tip (like food or transit). \
  Keep the response concise and lively - no long lists or generic info. Tell me like a real tour guide would.`

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Text-to-speech function
  const speakText = (text: string) => {
    if (!textToSpeechEnabled || !text.trim()) return

    // Check if speech synthesis is available
    if (!('speechSynthesis' in window)) {
      console.warn('Speech synthesis not supported')
      return
    }

    // Cancel any ongoing speech
    window.speechSynthesis.cancel()

    // Small delay to ensure cancellation is processed
    setTimeout(() => {
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 1.0
      utterance.pitch = 1.0
      utterance.volume = 1.0
      
      // Clear ref when speech ends
      utterance.onend = () => {
        speechSynthesisRef.current = null
      }
      
      utterance.onerror = (event) => {
        console.error('Speech synthesis error:', event)
        speechSynthesisRef.current = null
      }
      
      speechSynthesisRef.current = utterance
      
      // For mobile browsers, ensure we're in a user interaction context
      try {
        window.speechSynthesis.speak(utterance)
      } catch (error) {
        console.error('Error speaking text:', error)
      }
    }, 100)
  }

  // Stop speech when component unmounts or text-to-speech is disabled
  useEffect(() => {
    if (!textToSpeechEnabled) {
      window.speechSynthesis.cancel()
    }
    return () => {
      window.speechSynthesis.cancel()
    }
  }, [textToSpeechEnabled])

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 192)}px`
    }
  }, [input])

  // Reset function for pull-to-refresh
  const resetChat = () => {
    setMessages([])
    setInput('')
    setIsLoading(false)
    setError(null)
    setShowInput(false)
    setShowGoButton(true)
    setHasStarted(false)
    setCurrentLocation(null)
    hasInitialized.current = false
    window.speechSynthesis.cancel()
  }

  // Pull-to-refresh handlers
  const handleTouchStart = (e: React.TouchEvent) => {
    // Only allow pull-to-refresh when chat has started
    if (showGoButton || !chatMessagesRef.current) return
    const scrollTop = chatMessagesRef.current.scrollTop
    touchStartY.current = e.touches[0].clientY
    touchStartScrollTop.current = scrollTop
    isPulling.current = false
  }

  const handleTouchMove = (e: React.TouchEvent) => {
    // Only allow pull-to-refresh when chat has started
    if (showGoButton || !chatMessagesRef.current) return
    
    // Only allow pull-to-refresh if we're at the top of the scroll
    if (chatMessagesRef.current.scrollTop > 0) {
      isPulling.current = false
      setPullDistance(0)
      return
    }

    const touchY = e.touches[0].clientY
    const deltaY = touchY - touchStartY.current

    // Only allow pulling down (positive deltaY)
    if (deltaY > 0) {
      isPulling.current = true
      const distance = Math.min(deltaY * 0.5, 100) // Cap at 100px with resistance
      setPullDistance(distance)
      // Prevent default scrolling when pulling to refresh
      if (distance > 10) {
        e.preventDefault()
      }
    }
  }

  const handleTouchEnd = () => {
    // Only allow pull-to-refresh when chat has started
    if (showGoButton) return
    
    if (isPulling.current && pullDistance > 50) {
      // Trigger refresh
      setIsRefreshing(true)
      resetChat()
      
      // Small delay to show refresh animation, then restart
      setTimeout(() => {
        setIsRefreshing(false)
        setPullDistance(0)
        // Restart by calling handleGoClick after a brief moment
        setTimeout(() => {
          handleGoClick()
        }, 100)
      }, 500)
    } else {
      // Reset pull distance if not enough pull
      setPullDistance(0)
      isPulling.current = false
    }
  }

  // Get GPS location (but don't send initial message until Go is clicked)
  useEffect(() => {
    if (hasInitialized.current) return
    hasInitialized.current = true

    const getLocation = async () => {
      if (!navigator.geolocation) {
        setError('Geolocation is not supported by your browser')
        return
      }

      navigator.geolocation.getCurrentPosition(
        async (position) => {
          // Just get the position, don't geocode yet
          // Geocoding will happen when user clicks "Show me around"
        },
      (error) => {
        const errorMsg = `Error getting location: ${error.message}`
        setError(errorMsg)
        setIsLoading(false) // Stop loading on error
      },
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 0
        }
      )
    }

    getLocation()
  }, [])

  // Handle Go button click - send initial message
  const handleGoClick = async () => {
    if (hasStarted || isLoading) return
    
    setShowGoButton(false)
    setHasStarted(true)
    setIsLoading(true) // Show loading spinner immediately
    
    // Get current location
    if (!navigator.geolocation) {
      setError('Geolocation is not supported by your browser')
      setIsLoading(false)
      return
    }

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude } = position.coords
        
        try {
          // Get place name for display - catch errors gracefully
          let placeName: string
          try {
            placeName = await getPlaceName(latitude, longitude)
          } catch (err) {
            console.warn('Error getting place name, trying general location:', err)
            placeName = 'Unknown location'
          }
          
          // If location is unknown, try a more general location
          if (placeName === 'Unknown location' || placeName.toLowerCase().includes('unknown')) {
            try {
              placeName = await getPlaceName(latitude, longitude, true) // Try more general
            } catch (err) {
              console.warn('Error getting general place name:', err)
              placeName = 'Unknown location'
            }
          }
          
          // Store location for display (human-readable only, no GPS/Plus Code)
          const locationDisplay = placeName !== 'Unknown location' 
            ? placeName
            : null // Don't show location if unknown
          setCurrentLocation(locationDisplay)
          
          // Store original placeName for later use
          const originalPlaceName = placeName !== 'Unknown location' && !placeName.toLowerCase().includes('unknown') && !placeName.startsWith('Location at coordinates')
            ? placeName
            : null
          
          // Get coordinates and Plus Code for the message
          const plusCode = coordinatesToPlusCode(latitude, longitude)
          
          // Always include the human-readable place name from geolocation library in the initial message
          // Format includes location details and tour guide instructions summary
          // Only use placeName if it's a valid place name (not "Unknown location" or starting with "Location at")
          const hasValidPlaceName = placeName !== 'Unknown location' && 
                                   !placeName.toLowerCase().includes('unknown') && 
                                   !placeName.startsWith('Location at coordinates')
          
          const locationPart = hasValidPlaceName
            ? `I am currently at ${placeName} (coordinates: ${latitude.toFixed(6)}, ${longitude.toFixed(6)}, Plus Code: ${plusCode}).`
            : `I am currently at coordinates ${latitude.toFixed(6)}, ${longitude.toFixed(6)} (Plus Code: ${plusCode}).`
          
          const initialMessage = locationPart + TOUR_GUIDE_INSTRUCTIONS
          
          const userMessage: ChatMessage = {
            role: 'user',
            content: initialMessage,
            isHidden: true,
          }

          setMessages([userMessage])
          setIsLoading(true)
          setError(null)

          const apiMessages: Message[] = [
            { role: 'user', content: initialMessage },
          ]

          const assistantMessage: ChatMessage = {
            role: 'assistant',
            content: '',
            isStreaming: true,
          }
          setMessages([userMessage, assistantMessage])

          try {
            let fullContent = ''
            for await (const chunk of streamChatCompletion(apiMessages)) {
              fullContent += chunk
              setMessages((prev) => {
                const newMessages = [...prev]
                const lastMessage = newMessages[newMessages.length - 1]
                if (lastMessage && lastMessage.role === 'assistant') {
                  lastMessage.content = fullContent
                }
                return newMessages
              })
            }

            setMessages((prev) => {
              const newMessages = [...prev]
              const lastMessage = newMessages[newMessages.length - 1]
              if (lastMessage && lastMessage.role === 'assistant') {
                lastMessage.isStreaming = false
              }
              return newMessages
            })
            
            // Check if model indicates location is unknown and retry with more general location
            // This checks the model's response for patterns indicating it doesn't know the location
            const locationUnknownPatterns = [
              /don't know.*location/i,
              /unknown location/i,
              /can't.*tell.*about.*location/i,
              /location.*not.*found/i,
              /unable.*identify.*location/i,
              /don't have.*information.*location/i,
              /cannot.*identify.*location/i
            ]
            
            const isLocationUnknown = locationUnknownPatterns.some(pattern => 
              pattern.test(fullContent)
            )
            
            // If model says location is unknown, try with a more general location
            // How it works: We check the model's response text for patterns indicating
            // it doesn't know the location. If found, we fetch a more general location
            // (city/region level instead of specific address) and ask again.
            if (isLocationUnknown && originalPlaceName !== 'Unknown location') {
              try {
                const generalPlaceName = await getPlaceName(latitude, longitude, true)
                if (generalPlaceName !== 'Unknown location' && generalPlaceName !== originalPlaceName) {
                  // Update location display (human-readable only)
                  setCurrentLocation(generalPlaceName)
                  
                  // Send a follow-up message asking about the general location
                  const followUpMessage = `Actually, I'm in ${generalPlaceName}. Can you tell me about this area?`
                  const followUpUserMessage: ChatMessage = {
                    role: 'user',
                    content: followUpMessage,
                  }
                  
                  const updatedApiMessages: Message[] = [
                    { role: 'system', content: `You are an extremely knowledgeable, passionate tour guide with deep expertise in architecture, history, culture, geography, indigenous peoples, local food, transit, and practical logistics. Always respond in English only.

When providing tour information, include:
- Personal anecdotes ("The first time I came here...", "I always tell visitors...")
- Sensory details (smells, sounds, textures)
- Specific weather/timing with seasonal context
- Indigenous acknowledgment (original peoples, languages, stories)
- Food recommendations with prices and insider tips (distinguish tourist traps from authentic spots)
- Transit specifics (bus numbers, metro lines, walking times, costs)
- Natural next steps with distances and reasons
- Humor, warmth, and genuine enthusiasm
- Architecture/history depth with specific dates and details
- Local wildlife/plants when relevant

Structure your response flexibly (not formulaic): compelling opening, weather/seasonal context, historical/cultural depth, practical logistics, food recommendations, insider tips, next steps, and inviting closing. Keep responses shorter and punchier—cut fluff, keep passion. Be authentic: if you don't know specific details, focus on general cultural context and practical tips rather than making things up.` },
                    ...apiMessages.slice(1), // Skip the system message that's already there
                    { role: 'assistant', content: fullContent },
                    { role: 'user', content: followUpMessage },
                  ]
                  
                  const followUpAssistantMessage: ChatMessage = {
                    role: 'assistant',
                    content: '',
                    isStreaming: true,
                  }
                  
                  setMessages((prev) => [...prev, followUpUserMessage, followUpAssistantMessage])
                  
                  let followUpContent = ''
                  for await (const chunk of streamChatCompletion(updatedApiMessages)) {
                    followUpContent += chunk
                    setMessages((prev) => {
                      const newMessages = [...prev]
                      const lastMessage = newMessages[newMessages.length - 1]
                      if (lastMessage && lastMessage.role === 'assistant') {
                        lastMessage.content = followUpContent
                      }
                      return newMessages
                    })
                  }
                  
                  setMessages((prev) => {
                    const newMessages = [...prev]
                    const lastMessage = newMessages[newMessages.length - 1]
                    if (lastMessage && lastMessage.role === 'assistant') {
                      lastMessage.isStreaming = false
                    }
                    return newMessages
                  })
                  
                  fullContent = followUpContent // Update for TTS
                }
              } catch (err) {
                console.error('Error getting general location:', err)
              }
            }
            
            // Show input form after first response
            setShowInput(true)
            
            // Speak the assistant's response if text-to-speech is enabled
            // Use setTimeout to ensure it's after user interaction (required for mobile browsers)
            if (textToSpeechEnabled) {
              setTimeout(() => {
                speakText(fullContent)
              }, 500)
            }
          } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred')
            setMessages([userMessage])
          } finally {
            setIsLoading(false)
          }
        } catch (err) {
          // If geocoding fails, still try to get place name from geolocation library
          const plusCode = coordinatesToPlusCode(latitude, longitude)
          
          // Try to get place name from geolocation library
          let placeName = await getPlaceName(latitude, longitude, true)
          
          // Validate place name
          const hasValidPlaceName = placeName !== 'Unknown location' && 
                                   !placeName.toLowerCase().includes('unknown') && 
                                   !placeName.startsWith('Location at coordinates')
          
          // Store location for display
          const locationDisplay = hasValidPlaceName ? placeName : null
          setCurrentLocation(locationDisplay)
          
          // Always include the human-readable place name from geolocation library in the initial message
          const locationPart = hasValidPlaceName
            ? `I am currently at ${placeName} (coordinates: ${latitude.toFixed(6)}, ${longitude.toFixed(6)}, Plus Code: ${plusCode}).`
            : `I am currently at coordinates ${latitude.toFixed(6)}, ${longitude.toFixed(6)} (Plus Code: ${plusCode}).`
          
          const initialMessage = locationPart + TOUR_GUIDE_INSTRUCTIONS
          
          const userMessage: ChatMessage = {
            role: 'user',
            content: initialMessage,
            isHidden: true,
          }

          setMessages([userMessage])
          setIsLoading(true)
          setError(null)

          const apiMessages: Message[] = [
            { role: 'system', content: `You are an extremely knowledgeable, passionate tour guide with deep expertise in architecture, history, culture, geography, indigenous peoples, local food, transit, and practical logistics. Always respond in English only.

When providing tour information, include:
- Personal anecdotes ("The first time I came here...", "I always tell visitors...")
- Sensory details (smells, sounds, textures)
- Specific weather/timing with seasonal context
- Indigenous acknowledgment (original peoples, languages, stories)
- Food recommendations with prices and insider tips (distinguish tourist traps from authentic spots)
- Transit specifics (bus numbers, metro lines, walking times, costs)
- Natural next steps with distances and reasons
- Humor, warmth, and genuine enthusiasm
- Architecture/history depth with specific dates and details
- Local wildlife/plants when relevant

Structure your response flexibly (not formulaic): compelling opening, weather/seasonal context, historical/cultural depth, practical logistics, food recommendations, insider tips, next steps, and inviting closing. Keep responses shorter and punchier—cut fluff, keep passion. Be authentic: if you don't know specific details, focus on general cultural context and practical tips rather than making things up.` },
            { role: 'user', content: initialMessage },
          ]

          const assistantMessage: ChatMessage = {
            role: 'assistant',
            content: '',
            isStreaming: true,
          }
          setMessages([userMessage, assistantMessage])

          try {
            let fullContent = ''
            for await (const chunk of streamChatCompletion(apiMessages)) {
              fullContent += chunk
              setMessages((prev) => {
                const newMessages = [...prev]
                const lastMessage = newMessages[newMessages.length - 1]
                if (lastMessage && lastMessage.role === 'assistant') {
                  lastMessage.content = fullContent
                }
                return newMessages
              })
            }

            setMessages((prev) => {
              const newMessages = [...prev]
              const lastMessage = newMessages[newMessages.length - 1]
              if (lastMessage && lastMessage.role === 'assistant') {
                lastMessage.isStreaming = false
              }
              return newMessages
            })
            
            setShowInput(true)
            
            // Speak the assistant's response if text-to-speech is enabled
            if (textToSpeechEnabled) {
              setTimeout(() => {
                speakText(fullContent)
              }, 500)
            }
          } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred')
            setMessages([userMessage])
          } finally {
            setIsLoading(false)
          }
        }
      },
      (error) => {
        const errorMsg = `Error getting location: ${error.message}`
        setError(errorMsg)
        setIsLoading(false) // Stop loading on error
        setHasStarted(false)
        setShowGoButton(true)
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 0
      }
    )
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!input.trim() || isLoading) {
      return
    }

    const userMessage: ChatMessage = {
      role: 'user',
      content: input.trim(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setIsLoading(true)
    setError(null)

    // Build message history for API
    // Include system message to ensure English-only responses
    const systemMessage: Message = { role: 'system', content: `You are an extremely knowledgeable, passionate tour guide with deep expertise in architecture, history, culture, geography, indigenous peoples, local food, transit, and practical logistics. Always respond in English only.

When providing tour information, include:
- Personal anecdotes ("The first time I came here...", "I always tell visitors...")
- Sensory details (smells, sounds, textures)
- Specific weather/timing with seasonal context
- Indigenous acknowledgment (original peoples, languages, stories)
- Food recommendations with prices and insider tips (distinguish tourist traps from authentic spots)
- Transit specifics (bus numbers, metro lines, walking times, costs)
- Natural next steps with distances and reasons
- Humor, warmth, and genuine enthusiasm
- Architecture/history depth with specific dates and details
- Local wildlife/plants when relevant

Structure your response flexibly (not formulaic): compelling opening, weather/seasonal context, historical/cultural depth, practical logistics, food recommendations, insider tips, next steps, and inviting closing. Keep responses shorter and punchier—cut fluff, keep passion. Be authentic: if you don't know specific details, focus on general cultural context and practical tips rather than making things up.` }
    const apiMessages: Message[] = [
      systemMessage,
      ...messages.map((m) => ({ role: m.role, content: m.content })),
      { role: 'user', content: userMessage.content },
    ]

    // Add streaming assistant message
    const assistantMessage: ChatMessage = {
      role: 'assistant',
      content: '',
      isStreaming: true,
    }
    setMessages((prev) => [...prev, assistantMessage])

    try {
      let fullContent = ''
      for await (const chunk of streamChatCompletion(apiMessages)) {
        fullContent += chunk
        setMessages((prev) => {
          const newMessages = [...prev]
          const lastMessage = newMessages[newMessages.length - 1]
          if (lastMessage && lastMessage.role === 'assistant') {
            lastMessage.content = fullContent
          }
          return newMessages
        })
      }

      // Mark streaming as complete
      setMessages((prev) => {
        const newMessages = [...prev]
        const lastMessage = newMessages[newMessages.length - 1]
        if (lastMessage && lastMessage.role === 'assistant') {
          lastMessage.isStreaming = false
        }
        return newMessages
      })
      
      // Show input form after first response if not already shown
      if (!showInput) {
        setShowInput(true)
      }
      
      // Speak the assistant's response if text-to-speech is enabled
      if (textToSpeechEnabled) {
        setTimeout(() => {
          speakText(fullContent)
        }, 500)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
      setMessages((prev) => prev.slice(0, -1)) // Remove failed message
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="chat-container">
      {/* Text-to-Speech Toggle - Top right (later screen) */}
      {!showGoButton && (
        <div className="tts-toggle-container">
          <label className="tts-toggle">
            <input
              type="checkbox"
              checked={textToSpeechEnabled}
              onChange={(e) => setTextToSpeechEnabled(e.target.checked)}
            />
            <span className="tts-toggle-label">
              Speak aloud
            </span>
          </label>
        </div>
      )}

      {/* Location Heading */}
      {currentLocation && !showGoButton && (
        <div className="location-heading">
          <h2>{currentLocation}</h2>
        </div>
      )}

      <div 
        className="chat-messages"
        ref={chatMessagesRef}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
        style={{
          transform: pullDistance > 0 ? `translateY(${pullDistance}px)` : undefined,
          transition: isRefreshing ? 'transform 0.3s ease-out' : undefined,
        }}
      >
        {!showGoButton && pullDistance > 0 && (
          <div className="pull-to-refresh-indicator" style={{ opacity: Math.min(pullDistance / 50, 1) }}>
            {pullDistance > 50 ? (
              <span>Release to refresh</span>
            ) : (
              <span>Pull to refresh</span>
            )}
          </div>
        )}
        {showGoButton && (
          <div className="welcome-message">
            <h1 className="welcome-title">Tour Guide</h1>
            <button 
              className="go-button"
              onClick={handleGoClick}
              disabled={isLoading}
            >
              Show me around
            </button>
            {/* Text-to-Speech Toggle - Under button (initial screen) */}
            <div className="tts-toggle-container-initial">
              <label className="tts-toggle">
                <input
                  type="checkbox"
                  checked={textToSpeechEnabled}
                  onChange={(e) => setTextToSpeechEnabled(e.target.checked)}
                />
                <span className="tts-toggle-label">
                  Speak aloud
                </span>
              </label>
            </div>
          </div>
        )}
        {!showGoButton && isLoading && messages.length === 0 && (
          <div className="loading-container">
            <div className="spinner"></div>
            <p className="loading-text">Getting your location and preparing your tour...</p>
          </div>
        )}
        {messages.map((message, index) => {
          // Skip rendering hidden messages
          if (message.isHidden) return null
          
          return (
            <div
              key={index}
              className={`message ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}
            >
              <div className="message-content">
                {message.content}
                {message.isStreaming && <span className="cursor">▊</span>}
              </div>
            </div>
          )
        })}
        {error && (
          <div className="error-message">
            Error: {error}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {showInput && (
        <form className="chat-input-form" onSubmit={handleSubmit}>
        <textarea
          ref={textareaRef}
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              handleSubmit(e)
            }
          }}
          placeholder="Type your message..."
          rows={1}
          disabled={isLoading}
        />
        <button
          type="submit"
          className="submit-button"
          disabled={isLoading || !input.trim()}
        >
          Go
        </button>
      </form>
      )}
    </div>
  )
}

