# Offline Tour Guide Frontend

A React-based mobile frontend for the Offline Tour Guide application.

## Features

- ChatGPT-style chat interface
- Streaming responses from vLLM API
- Mobile-optimized design
- Smooth animations and fade-in effects
- Automatic GPS location detection
- Auto-sends initial message with your coordinates

## Setup

```bash
cd frontend
npm install
```

## Development

```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Accessing from Your Phone (Same WiFi)

**Important:** The app uses HTTPS (required for GPS location access). You'll need to accept the self-signed certificate warning in your browser.

1. Find your computer's local IP address:
   - Mac/Linux: `ifconfig | grep "inet " | grep -v 127.0.0.1`
   - Windows: `ipconfig` (look for IPv4 Address)
   - The dev server will also display the network URL when it starts

2. On your phone, open a browser and go to: `https://YOUR_IP_ADDRESS:3000`
   - Example: `https://192.168.1.100:3000`
   - **Note:** Use `https://` not `http://`

3. Accept the security warning about the self-signed certificate:
   - Chrome/Android: Tap "Advanced" → "Proceed to [IP address] (unsafe)"
   - Safari/iOS: Tap "Show Details" → "visit this website" → "Visit Website"

4. The app will automatically request GPS location permission and send your coordinates to the tour guide

## Build

```bash
npm run build
```

## API Configuration

The API endpoint is configured in `src/services/api.ts`. Update the `API_URL` constant to point to your vLLM server.

