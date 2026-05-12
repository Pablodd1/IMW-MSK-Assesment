// ============================================================================
// Boxer3D — YOLO11-pose + Three.js 3D Skeleton Renderer
// Real-time 3D pose tracking for MSK assessment
// Connects to Python pose engine via WebSocket
// ============================================================================

class Boxer3D {
    constructor(config = {}) {
        this.wsUrl = config.wsUrl || 'ws://localhost:8765';
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.ws = null;
        this.running = false;
        this.keypoints = [];
        this.persons = [];
        self.skeleton3D = null;

        // Configuration
        this.config = {
            camWidth: config.camWidth || 640,
            camHeight: config.camHeight || 480,
            sendInterval: config.sendInterval || 100, // ms between frames
            minConfidence: config.minConfidence || 0.5,
            smoothing: config.smoothing !== undefined ? config.smoothing : true,
            ...config
        };
    }

    // YOLO11 keypoint connections (COCO format)
    static SKELETON_CONNECTIONS = [
        [5, 6],  // shoulders
        [5, 7], [7, 9],   // left arm
        [6, 8], [8, 10],  // right arm
        [5, 11], [6, 12], // torso
        [11, 12],          // hips
        [11, 13], [13, 15], // left leg
        [12, 14], [14, 16], // right leg
        [0, 1], [0, 2],    // nose to eyes
        [1, 3], [2, 4],    // eyes to ears
    ];

    static KEYPOINT_COLORS = {
        0: '#ff4444',   // nose
        1: '#44ff44',   // left_eye
        2: '#4444ff',   // right_eye
        3: '#88ff88',   // left_ear
        4: '#8888ff',   // right_ear
        5: '#ffaa00',   // left_shoulder
        6: '#ff00aa',   // right_shoulder
        7: '#ff8800',   // left_elbow
        8: '#ff0088',   // right_elbow
        9: '#ffcc00',   // left_wrist
        10: '#ff00cc',  // right_wrist
        11: '#00ffaa',  // left_hip
        12: '#aa00ff',  // right_hip
        13: '#00ff88',  // left_knee
        14: '#8800ff',  // right_knee
        15: '#00ff66',  // left_ankle
        16: '#6600ff',  // right_ankle
    };

    // ─── Initialization ───

    async start(videoElementId = 'poseVideo', canvasId = 'poseOverlay') {
        if (this.running) return;
        this.running = true;

        // Setup video
        this.video = document.getElementById(videoElementId);
        if (!this.video) throw new Error(`Video element #${videoElementId} not found`);

        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: this.config.camWidth, height: this.config.camHeight, facingMode: 'user' }
        });
        this.video.srcObject = stream;
        await this.video.play();

        // Setup 2D overlay canvas
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            this.canvas = document.createElement('canvas');
            this.canvas.id = canvasId;
            this.canvas.style.position = 'absolute';
            this.canvas.style.top = '0';
            this.canvas.style.left = '0';
            this.canvas.style.pointerEvents = 'none';
            this.video.parentElement.appendChild(this.canvas);
        }
        this.canvas.width = this.config.camWidth;
        this.canvas.height = this.config.camHeight;
        this.ctx = this.canvas.getContext('2d');

        // Connect WebSocket
        await this.connectWebSocket();

        // Start frame loop
        this.frameLoop();
    }

    connectWebSocket() {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.wsUrl);
            this.ws.onopen = () => {
                console.log('[Boxer3D] WebSocket connected');
                resolve();
            };
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'skeleton') {
                        this.persons = data.persons || [];
                        this.keypoints = this.persons[0]?.keypoints || [];
                        this.draw2DOverlay();
                        this.dispatchEvent(new CustomEvent('skeleton', {
                            detail: { persons: this.persons, fps: data.fps }
                        }));
                    }
                } catch (e) { /* ignore parse errors */ }
            };
            this.ws.onerror = (err) => {
                console.warn('[Boxer3D] WS error, retrying in 2s...');
                setTimeout(() => this.connectWebSocket(), 2000);
            };
            this.ws.onclose = () => {
                if (this.running) {
                    console.log('[Boxer3D] WS closed, reconnecting...');
                    setTimeout(() => this.connectWebSocket(), 2000);
                }
            };
        });
    }

    // ─── Frame Loop ───

    frameLoop() {
        if (!this.running || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            if (this.running) setTimeout(() => this.frameLoop(), 500);
            return;
        }

        // Capture frame
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.config.camWidth;
        tempCanvas.height = this.config.camHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(this.video, 0, 0);
        const jpeg = tempCanvas.toDataURL('image/jpeg', 0.6).split(',')[1];

        // Send to pose engine
        this.ws.send(JSON.stringify({
            cmd: 'frame',
            jpeg: jpeg,
            // LiDAR depth data would go here when available
            depth: null
        }));

        setTimeout(() => this.frameLoop(), this.config.sendInterval);
    }

    // ─── 2D Overlay Drawing ───

    draw2DOverlay() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        if (!this.keypoints.length) return;

        // Draw connections
        for (const [i, j] of Boxer3D.SKELETON_CONNECTIONS) {
            const a = this.keypoints[i];
            const b = this.keypoints[j];
            if (a && b && a.confidence > this.config.minConfidence && b.confidence > this.config.minConfidence) {
                ctx.beginPath();
                ctx.moveTo(a.x * this.canvas.width, a.y * this.canvas.height);
                ctx.lineTo(b.x * this.canvas.width, b.y * this.canvas.height);
                ctx.strokeStyle = '#00ff88';
                ctx.lineWidth = 3;
                ctx.stroke();
            }
        }

        // Draw keypoints
        for (const kp of this.keypoints) {
            if (kp.confidence > this.config.minConfidence) {
                const x = kp.x * this.canvas.width;
                const y = kp.y * this.canvas.height;
                const color = Boxer3D.KEYPOINT_COLORS[kp.id] || '#ffffff';

                ctx.beginPath();
                ctx.arc(x, y, 5, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1;
                ctx.stroke();

                // Show depth if available
                if (kp.depth_valid) {
                    ctx.fillStyle = 'rgba(0,255,0,0.7)';
                    ctx.font = '10px monospace';
                    ctx.fillText(`${kp.z.toFixed(2)}m`, x + 8, y - 8);
                }
            }
        }
    }

    // ─── 3D Scene (Three.js) ───

    init3DScene(containerId = 'pose3dContainer') {
        const container = document.getElementById(containerId);
        if (!container) {
            console.warn('[Boxer3D] 3D container not found, creating...');
            return;
        }

        const THREE = window.THREE;
        if (!THREE) {
            console.warn('[Boxer3D] THREE not loaded, skipping 3D scene');
            return;
        }

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a1a);

        const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 10);
        camera.position.set(0, 0.5, 2.5);
        camera.lookAt(0, 0.5, 0);

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        // Lighting
        const ambient = new THREE.AmbientLight(0x404060);
        scene.add(ambient);
        const dir = new THREE.DirectionalLight(0xffffff, 1);
        dir.position.set(1, 2, 1);
        scene.add(dir);

        // Ground grid
        const grid = new THREE.GridHelper(2, 20, 0x4444ff, 0x222244);
        grid.position.y = -0.3;
        scene.add(grid);

        // Skeleton group
        const skeletonGroup = new THREE.Group();
        scene.add(skeletonGroup);

        // Joints (spheres) and bones (cylinders)
        const jointGeo = new THREE.SphereGeometry(0.02, 8, 8);
        const joints = {};
        const bones = {};

        this.skeleton3D = { scene, camera, renderer, skeletonGroup, joints, bones, container };

        // Listen for skeleton updates
        this.addEventListener('skeleton', (e) => {
            this.update3DSkeleton(e.detail.persons[0]?.keypoints || []);
        });

        // Animation loop
        const animate = () => {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        };
        animate();

        // Resize handler
        window.addEventListener('resize', () => {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        });
    }

    update3DSkeleton(keypoints) {
        if (!this.skeleton3D) return;
        const { skeletonGroup, joints, bones } = this.skeleton3D;
        const THREE = window.THREE;

        // Clear old
        while (skeletonGroup.children.length) {
            skeletonGroup.remove(skeletonGroup.children[0]);
        }

        if (!keypoints.length) return;

        // Create joint spheres
        const jointMap = {};
        for (const kp of keypoints) {
            if (kp.confidence < this.config.minConfidence) continue;
            const color = Boxer3D.KEYPOINT_COLORS[kp.id] || '#ffffff';
            const mat = new THREE.MeshStandardMaterial({
                color: color,
                emissive: color,
                emissiveIntensity: 0.3
            });
            const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.025, 8, 8), mat);
            // Map normalized 2D to 3D space
            mesh.position.set(
                (kp.x - 0.5) * 1.5,
                1.0 - kp.y,
                kp.z || 0
            );
            skeletonGroup.add(mesh);
            jointMap[kp.id] = mesh.position;
        }

        // Draw bones between connected joints
        for (const [i, j] of Boxer3D.SKELETON_CONNECTIONS) {
            const a = jointMap[i];
            const b = jointMap[j];
            if (!a || !b) continue;

            const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
            const dir = new THREE.Vector3().subVectors(b, a);
            const len = dir.length();
            dir.normalize();

            const boneMat = new THREE.MeshStandardMaterial({
                color: 0x00ff88,
                emissive: 0x00ff88,
                emissiveIntensity: 0.15,
                transparent: true,
                opacity: 0.7
            });
            const bone = new THREE.Mesh(
                new THREE.CylinderGeometry(0.008, 0.008, len, 4),
                boneMat
            );
            bone.position.copy(mid);
            bone.quaternion.setFromUnitVectors(
                new THREE.Vector3(0, 1, 0),
                dir
            );
            skeletonGroup.add(bone);
        }

        // Add head (slightly larger sphere at nose)
        if (jointMap[0]) {
            const headMat = new THREE.MeshStandardMaterial({
                color: 0x44aaff,
                emissive: 0x44aaff,
                emissiveIntensity: 0.2,
                transparent: true,
                opacity: 0.4
            });
            const head = new THREE.Mesh(new THREE.SphereGeometry(0.08, 12, 12), headMat);
            head.position.copy(jointMap[0]);
            head.position.y += 0.02;
            skeletonGroup.add(head);
        }
    }

    // ─── Cleanup ───

    stop() {
        this.running = false;
        if (this.ws) this.ws.close();
        if (this.video && this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(t => t.stop());
        }
    }
}

// Make available globally
window.Boxer3D = Boxer3D;
