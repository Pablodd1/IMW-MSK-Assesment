// Femto Mega WebSocket Connection
const IMW_FEMTO_DEFAULT_CONFIG = {
  poseEngineUrl: 'wss://pablodd1--pose-engine-ws-serve.modal.run/ws',
  supabaseUrl: '',
  demoMode: false,
  voiceEnabled: true
};

function getIMWFemtoConfig() {
  return { ...IMW_FEMTO_DEFAULT_CONFIG, ...(window.IMW_CONFIG || {}) };
}

class FemtoMegaClient {
  constructor(url = getIMWFemtoConfig().poseEngineUrl) {
    this.url = url;
    this.ws = null;
    this.onSkeletonData = null;
    this.onDepthFrame = null;
    this.onStatus = null;
    this.connected = false;
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('✅ Connected to Femto Mega camera');
        this.connected = true;
        if (this.onStatus) this.onStatus({ connected: true, url: this.url });
        resolve();
      };

      this.ws.onerror = (error) => {
        console.error('❌ Femto Mega connection failed:', error);
        this.connected = false;
        if (this.onStatus) this.onStatus({ connected: false, error, url: this.url });
        reject(error);
      };

      this.ws.onmessage = (event) => {
        const message = JSON.parse(event.data);

        if ((message.type === 'depth_frame' || message.depth || message.depth_frame) && this.onDepthFrame) {
          this.onDepthFrame(message.depth || message.depth_frame || message.data || message);
        }

        if ((message.type === 'skeleton_data' || message.type === 'skeleton' || message.persons || message.keypoints || message.landmarks) && this.onSkeletonData) {
          this.onSkeletonData(message.data || message);
        }
      };

      this.ws.onclose = () => {
        console.log('🔌 Femto Mega disconnected');
        this.connected = false;
        if (this.onStatus) this.onStatus({ connected: false, url: this.url });
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

  requestDepthFrame() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ command: 'depth_frame' }));
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
    }
  }
}
