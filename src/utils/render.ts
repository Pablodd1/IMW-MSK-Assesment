// Real-time Joint Tracking Module - Three.js Integration
import * as THREE from 'three';

export class JointRenderer {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private joints: Map<string, THREE.Mesh> = new Map();
  private lines: THREE.Line[] = [];
  private skeletonGroup: THREE.Group;

  constructor(container: HTMLElement) {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1a2e);

    // Camera setup for medical visualization
    this.camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    this.camera.position.z = 5;

    // WebGL renderer with high-performance settings
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(this.renderer.domElement);

    // Skeleton group for efficient updates
    this.skeletonGroup = new THREE.Group();
    this.scene.add(this.skeletonGroup);

    // Lighting for medical visualization
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    this.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 5, 5);
    this.scene.add(directionalLight);

    // Initialize skeleton joints
    this.initializeSkeleton();
  }

  private initializeSkeleton() {
    const jointMaterial = new THREE.MeshPhongMaterial({ color: 0x00ff00 });
    const boneMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 });

    const jointPositions = [
      'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
      'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
      'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
      'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ];

    jointPositions.forEach(name => {
      const geometry = new THREE.SphereGeometry(0.05, 16, 16);
      const mesh = new THREE.Mesh(geometry, jointMaterial);
      mesh.visible = false;
      this.joints.set(name, mesh);
      this.skeletonGroup.add(mesh);
    });
  }

  updateSkeleton(landmarks: Record<string, { x: number; y: number; z: number }>) {
    const scaleFactor = 10;

    // Batch update positions
    this.joints.forEach((mesh, name) => {
      const landmark = landmarks[name];
      if (landmark) {
        mesh.position.set(
          (landmark.x - 0.5) * scaleFactor,
          -(landmark.y - 0.5) * scaleFactor,
          landmark.z * scaleFactor
        );
        mesh.visible = true;
      }
    });

    this.drawBones(landmarks, scaleFactor);
  }

  private drawBones(landmarks: Record<string, any>, scaleFactor: number) {
    // Clear previous lines
    this.lines.forEach(line => this.skeletonGroup.remove(line));
    this.lines = [];

    const bonePairs = [
      ['left_shoulder', 'left_elbow'],
      ['left_elbow', 'left_wrist'],
      ['right_shoulder', 'right_elbow'],
      ['right_elbow', 'right_wrist'],
      ['left_shoulder', 'right_shoulder'],
      ['left_hip', 'right_hip'],
      ['left_shoulder', 'left_hip'],
      ['right_shoulder', 'right_hip'],
      ['left_hip', 'left_knee'],
      ['left_knee', 'left_ankle'],
      ['right_hip', 'right_knee'],
      ['right_knee', 'right_ankle']
    ];

    const boneMaterial = new THREE.LineBasicMaterial({ color: 0x00ffff });

    bonePairs.forEach(([start, end]) => {
      if (landmarks[start] && landmarks[end]) {
        const geometry = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(
            (landmarks[start].x - 0.5) * scaleFactor,
            -(landmarks[start].y - 0.5) * scaleFactor,
            landmarks[start].z * scaleFactor
          ),
          new THREE.Vector3(
            (landmarks[end].x - 0.5) * scaleFactor,
            -(landmarks[end].y - 0.5) * scaleFactor,
            landmarks[end].z * scaleFactor
          )
        ]);
        const line = new THREE.Line(geometry, boneMaterial);
        this.skeletonGroup.add(line);
        this.lines.push(line);
      }
    });
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  resize(width: number, height: number) {
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }
}

// Camera Manager for multiple camera types
export class CameraManager {
  private stream: MediaStream | null = null;
  private videoElement: HTMLVideoElement;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor() {
    this.videoElement = document.createElement('video');
    this.videoElement.setAttribute('playsinline', '');
    this.videoElement.setAttribute('autoplay', '');
    
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d')!;
  }

  async initialize(deviceId?: string) {
    const constraints: MediaStreamConstraints = {
      video: deviceId ? { deviceId: { exact: deviceId } } : {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user'
      },
      audio: false
    };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.videoElement.srcObject = this.stream;
      await this.videoElement.play();
      this.canvas.width = this.videoElement.videoWidth || 1280;
      this.canvas.height = this.videoElement.videoHeight || 720;
      return true;
    } catch (error) {
      console.error('Camera initialization failed:', error);
      return false;
    }
  }

  async listDevices() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter(d => d.kind === 'videoinput');
  }

  captureFrame(): ImageData {
    this.ctx.drawImage(this.videoElement, 0, 0);
    return this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
  }

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
  }
}
