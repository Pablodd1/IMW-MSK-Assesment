// Femto Mega WebSocket Connection
class FemtoMegaClient {
  constructor(url = 'ws://localhost:8765') {
    this.url = url;
    this.ws = null;
    this.onSkeletonData = null;
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('✅ Connected to Femto Mega camera');
        resolve();
      };

      this.ws.onerror = (error) => {
        console.error('❌ Femto Mega connection failed:', error);
        reject(error);
      };

      this.ws.onmessage = (event) => {
        const message = JSON.parse(event.data);

        if (message.type === 'skeleton_data' && this.onSkeletonData) {
          this.onSkeletonData(message.data);
        }
      };

      this.ws.onclose = () => {
        console.log('🔌 Femto Mega disconnected');
      };
    });
  }

  startRecording() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ command: 'start_recording' }));
    }
  }

  stopRecording() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ command: 'stop_recording' }));
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
    }
  }
}
