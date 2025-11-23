/**
 * Offline geocoding cache using IndexedDB
 * Caches street-level reverse geocoding results for Australia
 */

const DB_NAME = 'OfflineTourGuideGeocache'
const STORE_NAME = 'geocache'
const DB_VERSION = 1

// Australia bounding box (approximate)
const AUS_LAT_MIN = -44
const AUS_LAT_MAX = -10
const AUS_LON_MIN = 113
const AUS_LON_MAX = 154

interface CachedGeocode {
  lat: number
  lon: number
  address: string
  timestamp: number
}

let db: IDBDatabase | null = null
// In-memory cache for recently accessed addresses (avoids IndexedDB lookups)
const memoryCache = new Map<string, { address: string; timestamp: number }>()
const MEMORY_CACHE_MAX_AGE = 5 * 60 * 1000 // 5 minutes

async function openDB(): Promise<IDBDatabase> {
  if (db) return db

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)

    request.onerror = () => reject(request.error)
    request.onsuccess = () => {
      db = request.result
      resolve(db)
    }

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'key' })
        store.createIndex('timestamp', 'timestamp', { unique: false })
      }
    }
  })
}

function getCacheKey(lat: number, lon: number): string {
  // Round to ~100m precision for caching (4 decimal places)
  return `${lat.toFixed(4)},${lon.toFixed(4)}`
}

function isInAustralia(lat: number, lon: number): boolean {
  return lat >= AUS_LAT_MIN && lat <= AUS_LAT_MAX && 
         lon >= AUS_LON_MIN && lon <= AUS_LON_MAX
}

export async function getCachedAddress(lat: number, lon: number): Promise<string | null> {
  // Only cache Australia locations
  if (!isInAustralia(lat, lon)) {
    return null
  }

  const key = getCacheKey(lat, lon)
  
  // Check in-memory cache first (fastest)
  const memoryEntry = memoryCache.get(key)
  if (memoryEntry) {
    const age = Date.now() - memoryEntry.timestamp
    if (age < MEMORY_CACHE_MAX_AGE) {
      return memoryEntry.address
    } else {
      memoryCache.delete(key)
    }
  }

  try {
    const database = await openDB()

    return new Promise((resolve, reject) => {
      const transaction = database.transaction([STORE_NAME], 'readonly')
      const store = transaction.objectStore(STORE_NAME)
      const request = store.get(key)

      request.onerror = () => reject(request.error)
      request.onsuccess = () => {
        const result = request.result
        if (result && result.address) {
          // Check if cache is still valid (30 days)
          const age = Date.now() - result.timestamp
          if (age < 30 * 24 * 60 * 60 * 1000) {
            // Store in memory cache for faster subsequent access
            memoryCache.set(key, { address: result.address, timestamp: Date.now() })
            resolve(result.address)
          } else {
            resolve(null)
          }
        } else {
          resolve(null)
        }
      }
    })
  } catch (error) {
    console.error('Error reading from geocache:', error)
    return null
  }
}

export async function cacheAddress(lat: number, lon: number, address: string): Promise<void> {
  // Only cache Australia locations
  if (!isInAustralia(lat, lon)) {
    return
  }

  const key = getCacheKey(lat, lon)
  const timestamp = Date.now()
  
  // Update memory cache immediately (fast)
  memoryCache.set(key, { address, timestamp })

  try {
    const database = await openDB()

    return new Promise((resolve, reject) => {
      const transaction = database.transaction([STORE_NAME], 'readwrite')
      const store = transaction.objectStore(STORE_NAME)
      const request = store.put({
        key,
        lat,
        lon,
        address,
        timestamp
      })

      request.onerror = () => reject(request.error)
      request.onsuccess = () => resolve()
    })
  } catch (error) {
    console.error('Error caching address:', error)
  }
}

export async function clearOldCache(daysToKeep: number = 30): Promise<void> {
  try {
    const database = await openDB()
    const cutoffTime = Date.now() - (daysToKeep * 24 * 60 * 60 * 1000)

    return new Promise((resolve, reject) => {
      const transaction = database.transaction([STORE_NAME], 'readwrite')
      const store = transaction.objectStore(STORE_NAME)
      const index = store.index('timestamp')
      const request = index.openCursor(IDBKeyRange.upperBound(cutoffTime))

      request.onerror = () => reject(request.error)
      request.onsuccess = (event) => {
        const cursor = (event.target as IDBRequest<IDBCursorWithValue>).result
        if (cursor) {
          cursor.delete()
          cursor.continue()
        } else {
          resolve()
        }
      }
    })
  } catch (error) {
    console.error('Error clearing old cache:', error)
  }
}

