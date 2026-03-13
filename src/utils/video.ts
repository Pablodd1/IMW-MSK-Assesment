// Video capture and processing
import { CameraManager } from './render';

export class VideoProcessor {
  private camera: CameraManager;
  private processing: boolean = false;
  private frameCallback?: (data: ImageData) => void;

  constructor() {
    this.camera = new CameraManager();
  }

  async start(deviceId?: string) {
    const success = await this.camera.initialize(deviceId);
    if (!success) throw new Error('Failed to initialize camera');
    this.processing = true;
    this.processFrame();
    return success;
  }

  private processFrame() {
    if (!this.processing) return;

    const imageData = this.camera.captureFrame();
    if (this.frameCallback) {
      this.frameCallback(imageData);
    }

    requestAnimationFrame(() => this.processFrame());
  }

  onFrame(callback: (data: ImageData) => void) {
    this.frameCallback = callback;
  }

  stop() {
    this.processing = false;
    this.camera.stop();
  }
}

// Worker for offloading processing
export class ProcessingWorker {
  private worker: Worker | null = null;

  constructor() {
    const workerCode = `
      self.onmessage = function(e) {
        const { imageData, landmarks } = e.data;
        // Process frame data in worker thread
        // Return processed landmarks
        self.postMessage({ landmarks });
      };
    `;
    const blob = new Blob([workerCode], { type: 'application/javascript' });
    this.worker = new Worker(URL.createObjectURL(blob));
  }

  process(imageData: ImageData): Promise<any> {
    return new Promise((resolve) => {
      if (!this.worker) {
        resolve({});
        return;
      }
      this.worker.onmessage = (e) => resolve(e.data);
      this.worker.postMessage({ imageData });
    });
  }

  terminate() {
    this.worker?.terminate();
  }
}
