import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
// import basicSsl from '@vitejs/plugin-basic-ssl' // Temporarily disabled for tunnel

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    // basicSsl() // Disabled - tunnel provides HTTPS layer
  ],
  server: {
    port: 3000,
    host: '0.0.0.0', // Explicitly bind to all network interfaces
    strictPort: true,
    // Removed allowedHosts to allow any instatunnel subdomain (useful for random subdomains)
    hmr: {
      // Let Vite auto-detect the HMR host from the request
      // This works better with tunnels that may have dynamic hostnames
      protocol: 'wss', // Use secure WebSocket through tunnel
      clientPort: 443 // Tunnel uses HTTPS/WSS on port 443
    }
  }
})

