import { OpenLocationCode } from 'open-location-code'
// @ts-ignore - reverse-geocode may not have TypeScript definitions
import reverseGeocode from 'reverse-geocode'
import { getCachedAddress, cacheAddress } from './geocache'

/**
 * Convert GPS coordinates to Plus Code
 */
export function coordinatesToPlusCode(latitude: number, longitude: number): string {
  const olc = new OpenLocationCode()
  return olc.encode(latitude, longitude)
}

/**
 * Helper function to fetch from multiple proxies in parallel and return the first successful response
 */
async function fetchWithProxies(nominatimUrl: string, timeoutMs: number = 5000): Promise<Response> {
  const proxyServices = [
    `https://api.allorigins.win/raw?url=${encodeURIComponent(nominatimUrl)}`,
    `https://api.codetabs.com/v1/proxy?quest=${encodeURIComponent(nominatimUrl)}`
  ]
  
  // Try all proxies in parallel
  const promises = proxyServices.map(proxiedUrl => {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs)
    
    return fetch(proxiedUrl, {
      method: 'GET',
      mode: 'cors',
      headers: { 'Accept': 'application/json' },
      signal: controller.signal
    })
      .then(response => {
        clearTimeout(timeoutId)
        if (response.ok) {
          return response
        }
        throw new Error(`Proxy returned ${response.status}`)
      })
      .catch(err => {
        clearTimeout(timeoutId)
        throw err
      })
  })
  
  // Return the first successful response, or throw if all fail
  // Use a polyfill for Promise.any if not available
  const errors: Error[] = []
  for (const promise of promises) {
    try {
      return await promise
    } catch (err) {
      errors.push(err as Error)
    }
  }
  // Create a simple error if AggregateError is not available
  const error = new Error('All proxies failed')
  ;(error as any).errors = errors
  throw error
}

/**
 * Get detailed location information from GPS coordinates
 * Returns a human-readable description of the location
 * @param latitude - Latitude coordinate
 * @param longitude - Longitude coordinate
 * @param useGeneralLocation - If true, uses lower zoom level for more general location (city/region level)
 */
export async function getLocationDescription(latitude: number, longitude: number, useGeneralLocation: boolean = false): Promise<string> {
  const plusCode = coordinatesToPlusCode(latitude, longitude)
  
  // Check cache once at the start
  let cachedDesc: string | null = null
  try {
    cachedDesc = await getCachedAddress(latitude, longitude)
    if (cachedDesc) {
      return `${cachedDesc} (coordinates: ${latitude.toFixed(6)}, ${longitude.toFixed(6)}, Plus Code: ${plusCode})`
    }
  } catch (cacheError) {
    // Continue if cache check fails
  }

  try {
    // Try offline reverse geocoding first (US locations only)
    try {
      // For US locations, try reverse-geocode for offline support
      const isLikelyUS = latitude >= 24 && latitude <= 50 && longitude >= -125 && longitude <= -66
      
      if (isLikelyUS) {
        // @ts-ignore - reverse-geocode may not have TypeScript definitions
        const result = reverseGeocode.lookup(latitude, longitude, 'us')
        
        // Build human-readable description
        const parts: string[] = []
        
        // Note: reverse-geocode may have different property names
        // @ts-ignore
        if (result.neighborhood && !useGeneralLocation) {
          // @ts-ignore
          parts.push(result.neighborhood)
        }
        if (result.city) {
          parts.push(result.city)
        }
        if (result.state) {
          parts.push(result.state)
        }
        if (result.country) {
          parts.push(result.country)
        }
        
        if (parts.length > 0) {
          return `${parts.join(', ')} (coordinates: ${latitude.toFixed(6)}, ${longitude.toFixed(6)}, Plus Code: ${plusCode})`
        }
      }
    } catch (offlineError) {
      // Fall through to online API for detailed street-level addresses
      // Note: offline geocoding provides city-level accuracy, online API provides street-level
    }

    // Use online OpenStreetMap Nominatim API (works worldwide, including Australia)
    // This is the primary method for non-US locations
    const zoom = useGeneralLocation ? 10 : 18
    
    let data: any
    try {
      const nominatimUrl = `https://nominatim.openstreetmap.org/reverse?format=json&lat=${latitude}&lon=${longitude}&zoom=${zoom}&addressdetails=1`
      
      // Use parallel proxy fetching with shorter timeout
      const response = await fetchWithProxies(nominatimUrl, 6000) // 6 second timeout

      if (!response.ok) {
        throw new Error(`Geocoding failed: ${response.status} ${response.statusText}`)
      }

      // Parse response - some proxies wrap the JSON, others return it directly
      const responseText = await response.text()
      try {
        // Try parsing directly first
        const parsed = JSON.parse(responseText)
        // Check if it's wrapped (allorigins.get returns {contents: "...", status: {...}})
        if (parsed.contents) {
          // allorigins.get wraps the response
          data = JSON.parse(parsed.contents)
        } else {
          // Direct response
          data = parsed
        }
      } catch (parseError) {
        // If direct parse fails, try parsing as wrapped response
        try {
          const wrapped = JSON.parse(responseText)
          data = wrapped.contents ? JSON.parse(wrapped.contents) : wrapped
        } catch {
          throw new Error('Failed to parse geocoding response')
        }
      }
      
      if (data.error) {
        throw new Error(data.error)
      }
    } catch (fetchError: any) {
      // Check if it's an abort (timeout) or network error
      if (fetchError.name === 'AbortError') {
        console.warn('Geocoding request timed out')
      } else if (fetchError.message?.includes('Failed to fetch') || fetchError.message?.includes('NetworkError')) {
        console.warn('Network error during geocoding:', fetchError.message)
      }
      
      // If we have a cached value, use it (already checked at start)
      if (cachedDesc) {
        return `${cachedDesc} (coordinates: ${latitude.toFixed(6)}, ${longitude.toFixed(6)}, Plus Code: ${plusCode})`
      }
      // Re-throw the error to be handled by outer catch block
      throw fetchError
    }

    // Build human-readable location description
    const addressParts: string[] = []
    
    if (data.address) {
      if (useGeneralLocation) {
        // For general location, build from larger to smaller areas
        if (data.address.city) addressParts.push(data.address.city)
        else if (data.address.town) addressParts.push(data.address.town)
        else if (data.address.village) addressParts.push(data.address.village)
        
        if (data.address.state) addressParts.push(data.address.state)
        else if (data.address.region) addressParts.push(data.address.region)
        else if (data.address.county) addressParts.push(data.address.county)
        
        if (data.address.country) addressParts.push(data.address.country)
      } else {
        // For specific location, include more details
        if (data.address.tourism) addressParts.push(data.address.tourism)
        else if (data.address.attraction) addressParts.push(data.address.attraction)
        else if (data.address.historic) addressParts.push(data.address.historic)
        else if (data.address.amenity) addressParts.push(data.address.amenity)
        else if (data.address.building) addressParts.push(data.address.building)
        
        if (data.address.road) addressParts.push(data.address.road)
        if (data.address.suburb) addressParts.push(data.address.suburb)
        if (data.address.city) addressParts.push(data.address.city)
        else if (data.address.town) addressParts.push(data.address.town)
        else if (data.address.village) addressParts.push(data.address.village)
        
        if (data.address.state) addressParts.push(data.address.state)
        else if (data.address.region) addressParts.push(data.address.region)
        if (data.address.country) addressParts.push(data.address.country)
      }
    }
    
    // Build final description
    if (addressParts.length > 0) {
      return `${addressParts.join(', ')} (coordinates: ${latitude.toFixed(6)}, ${longitude.toFixed(6)}, Plus Code: ${plusCode})`
    }
    
    // Fallback to display_name if available
    if (data.display_name) {
      return `${data.display_name} (coordinates: ${latitude.toFixed(6)}, ${longitude.toFixed(6)}, Plus Code: ${plusCode})`
    }
    
    // Last resort: just coordinates and plus code
    return `Location at coordinates ${latitude.toFixed(6)}, ${longitude.toFixed(6)} (Plus Code: ${plusCode})`
  } catch (error) {
    // If we have a cached value, use it (already checked at start)
    if (cachedDesc) {
      return `${cachedDesc} (coordinates: ${latitude.toFixed(6)}, ${longitude.toFixed(6)}, Plus Code: ${plusCode})`
    }
    
    // Log error but don't crash - return a fallback
    console.warn('Error getting location description, using fallback:', error instanceof Error ? error.message : error)
    
    // Return coordinates and plus code as fallback
    return `Location at coordinates ${latitude.toFixed(6)}, ${longitude.toFixed(6)} (Plus Code: ${plusCode})`
  }
}

/**
 * Get place name from GPS coordinates using offline reverse geocoding
 * Falls back to online API if offline method fails
 * @param latitude - Latitude coordinate
 * @param longitude - Longitude coordinate
 * @param useGeneralLocation - If true, uses lower zoom level for more general location (city/region level)
 */
export async function getPlaceName(latitude: number, longitude: number, useGeneralLocation: boolean = false): Promise<string> {
  // Check cache once at the start
  let cachedAddress: string | null = null
  try {
    cachedAddress = await getCachedAddress(latitude, longitude)
    if (cachedAddress) {
      return cachedAddress
    }
  } catch (cacheError) {
    // Continue if cache check fails
  }
  
  try {

    // Try offline reverse geocoding first (US locations only)
    try {
      // For US locations, try reverse-geocode for offline support
      const isLikelyUS = latitude >= 24 && latitude <= 50 && longitude >= -125 && longitude <= -66
      
      if (isLikelyUS) {
        // @ts-ignore - reverse-geocode may not have TypeScript definitions
        const result = reverseGeocode.lookup(latitude, longitude, 'us')
        
        // Build place name from available fields
        const parts: string[] = []
        
        if (useGeneralLocation) {
          // For general location, prefer larger areas
          if (result.city) parts.push(result.city)
          if (result.state) parts.push(result.state)
          if (result.country) parts.push(result.country)
        } else {
          // For specific location, prefer more specific fields
          // @ts-ignore - neighborhood may not exist in type definition
          if (result.neighborhood) parts.push(result.neighborhood)
          if (result.city) parts.push(result.city)
          if (result.state) parts.push(result.state)
        }
        
        if (parts.length > 0) {
          return parts.join(', ')
        }
        
        // Fallback to city or state if available
        if (result.city) return result.city
        if (result.state) return result.state
        if (result.country) return result.country
      }
    } catch (offlineError) {
      // Silently fall through to online API if offline geocoding fails
    }

    // Use online OpenStreetMap Nominatim API for reverse geocoding (works worldwide, including Australia)
    // This is the primary method for non-US locations and fallback for US locations
    // Use zoom 18 for street-level detail, 10 for city-level
    const zoom = useGeneralLocation ? 10 : 18
    
    let data: any
    try {
      const nominatimUrl = `https://nominatim.openstreetmap.org/reverse?format=json&lat=${latitude}&lon=${longitude}&zoom=${zoom}&addressdetails=1`
      
      // Use parallel proxy fetching with shorter timeout
      const response = await fetchWithProxies(nominatimUrl, 6000) // 6 second timeout

      if (!response.ok) {
        throw new Error(`Geocoding failed: ${response.status} ${response.statusText}`)
      }

      // Parse response - some proxies wrap the JSON, others return it directly
      const responseText = await response.text()
      try {
        // Try parsing directly first
        const parsed = JSON.parse(responseText)
        // Check if it's wrapped (allorigins.get returns {contents: "...", status: {...}})
        if (parsed.contents) {
          // allorigins.get wraps the response
          data = JSON.parse(parsed.contents)
        } else {
          // Direct response
          data = parsed
        }
      } catch (parseError) {
        // If direct parse fails, try parsing as wrapped response
        try {
          const wrapped = JSON.parse(responseText)
          data = wrapped.contents ? JSON.parse(wrapped.contents) : wrapped
        } catch {
          throw new Error('Failed to parse geocoding response')
        }
      }
      
      if (data.error) {
        throw new Error(data.error)
      }
    } catch (fetchError: any) {
      // Check if it's an abort (timeout) or network error
      if (fetchError.name === 'AbortError') {
        console.warn('Geocoding request timed out')
      } else if (fetchError.message?.includes('Failed to fetch') || fetchError.message?.includes('NetworkError')) {
        console.warn('Network error during geocoding:', fetchError.message)
      }
      
      // If we have a cached value, use it (already checked at start)
      if (cachedAddress) {
        return cachedAddress
      }
      
      // Log the error for debugging
      console.warn('Geocoding API failed:', fetchError?.message || fetchError)
      
      // Re-throw to be handled by outer catch - it will return 'Unknown location' gracefully
      throw fetchError
    }

    // Build a full address string with multiple location identifiers
    if (data.address) {
      if (useGeneralLocation) {
        // For general location, prefer larger areas
        const placeName = 
          data.address.city ||
          data.address.town ||
          data.address.village ||
          data.address.municipality ||
          data.address.county ||
          data.address.state ||
          data.address.region ||
          data.address.country ||
          data.display_name

        return placeName || data.display_name || 'Unknown location'
      } else {
        // For specific location, build a full address: house_number, road, suburb, city, state, country
        const addressParts: string[] = []
        
        // Start with street address (house number + road)
        // Prioritize getting street-level data
        if (data.address.house_number) {
          addressParts.push(data.address.house_number)
        }
        if (data.address.road) {
          addressParts.push(data.address.road)
        }
        
        // If we don't have house_number or road, try to get any street-level identifier
        if (addressParts.length === 0) {
          // Try alternative street identifiers
          if (data.address.pedestrian) {
            addressParts.push(data.address.pedestrian)
          } else if (data.address.path) {
            addressParts.push(data.address.path)
          } else if (data.address.footway) {
            addressParts.push(data.address.footway)
          }
        }
        
        // Add suburb/neighborhood (important for street-level specificity)
        if (data.address.suburb) {
          addressParts.push(data.address.suburb)
        } else if (data.address.neighbourhood) {
          addressParts.push(data.address.neighbourhood)
        } else if (data.address.city_district) {
          addressParts.push(data.address.city_district)
        }
        
        // Add city/town
        if (data.address.city) {
          addressParts.push(data.address.city)
        } else if (data.address.town) {
          addressParts.push(data.address.town)
        } else if (data.address.village) {
          addressParts.push(data.address.village)
        } else if (data.address.municipality) {
          addressParts.push(data.address.municipality)
        }
        
        // Add state/region
        if (data.address.state) {
          addressParts.push(data.address.state)
        } else if (data.address.region) {
          addressParts.push(data.address.region)
        }
        
        // Add country
        if (data.address.country) {
          addressParts.push(data.address.country)
        }
        
        // If we have a meaningful address (at least 2 parts for specificity), cache it and return it
        if (addressParts.length >= 2) {
          const fullAddress = addressParts.join(', ')
          // Cache the result for offline use (Australia only)
          await cacheAddress(latitude, longitude, fullAddress)
          return fullAddress
        }
        
        // If we only have one part, try to get more context from display_name
        if (addressParts.length === 1 && data.display_name) {
          // Try to extract more specific info from display_name
          const displayParts = data.display_name.split(', ')
          // Use the first few parts of display_name for more context
          const enhancedAddress = displayParts.slice(0, Math.min(4, displayParts.length)).join(', ')
          if (enhancedAddress && enhancedAddress.length > addressParts[0].length) {
            await cacheAddress(latitude, longitude, enhancedAddress)
            return enhancedAddress
          }
        }
        
        // Fallback to named places if no street address
        const placeName = 
          data.address.tourism ||
          data.address.attraction ||
          data.address.historic ||
          data.address.amenity ||
          data.address.building ||
          data.address.road ||
          data.display_name

        const result = placeName || data.display_name || 'Unknown location'
        // Cache even fallback results for consistency
        if (result !== 'Unknown location') {
          await cacheAddress(latitude, longitude, result)
        }
        return result
      }
    }

    const result = data.display_name || 'Unknown location'
    // Cache the result for offline use (Australia only)
    if (result !== 'Unknown location') {
      await cacheAddress(latitude, longitude, result)
    }
    return result
  } catch (error) {
    console.error('Error getting place name:', error)
    // If we have a cached value, use it (already checked at start)
    if (cachedAddress) {
      return cachedAddress
    }
    // Return a fallback if geocoding fails
    return 'Unknown location'
  }
}

