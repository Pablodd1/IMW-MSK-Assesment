import asyncio
import json
import websockets
import time
import statistics
from datetime import datetime

NUM_CLIENTS = 50
NUM_MESSAGES = 50
SERVER_URI = "ws://localhost:8765"

async def client_task(client_id, results, barrier):
    try:
        async with websockets.connect(SERVER_URI) as websocket:
            # Wait for initial connection message
            await websocket.recv()

            # Wait for all clients to connect
            await barrier.wait()

            # Only client 0 starts streaming
            if client_id == 0:
                await websocket.send(json.dumps({"command": "start_streaming"}))

            count = 0
            while count < NUM_MESSAGES:
                msg = await websocket.recv()
                recv_time = time.time()
                try:
                    data = json.loads(msg)
                    if data.get('type') == 'skeleton':
                        skeleton = data.get('skeleton')
                        ts_str = skeleton.get('timestamp')
                        # Parse timestamp
                        # Handles ISO format like '2023-10-27T10:00:00.123456'
                        ts = datetime.fromisoformat(ts_str).timestamp()
                        latency = recv_time - ts
                        results[client_id].append(latency)
                        count += 1
                except Exception as e:
                    # Ignore non-skeleton messages or errors
                    pass
    except Exception as e:
        print(f"Client {client_id} error: {e}")

async def run_benchmark():
    results = [[] for _ in range(NUM_CLIENTS)]
    barrier = asyncio.Barrier(NUM_CLIENTS)
    tasks = []

    print(f"Connecting {NUM_CLIENTS} clients...")
    for i in range(NUM_CLIENTS):
        tasks.append(client_task(i, results, barrier))

    # Run until all tasks complete (collect NUM_MESSAGES each)
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        print(f"Benchmark error: {e}")

    # Analyze results
    all_latencies = []
    for r in results:
        all_latencies.extend(r)

    print(f"Total messages analyzed: {len(all_latencies)}")
    if not all_latencies:
        print("No messages received.")
        return

    # Use statistics module for standard library compatibility if numpy is missing
    # But since requirements say numpy is there, I'll stick to basic stats manually to be safe
    # or just use statistics module

    avg_latency = statistics.mean(all_latencies)
    try:
        # p95 approx
        sorted_lat = sorted(all_latencies)
        p95_idx = int(len(sorted_lat) * 0.95)
        p95_latency = sorted_lat[p95_idx]
        p99_idx = int(len(sorted_lat) * 0.99)
        p99_latency = sorted_lat[p99_idx]
        std_dev = statistics.stdev(all_latencies)
    except:
        p95_latency = 0
        p99_latency = 0
        std_dev = 0

    print(f"Average Latency: {avg_latency*1000:.2f} ms")
    print(f"P95 Latency: {p95_latency*1000:.2f} ms")
    print(f"P99 Latency: {p99_latency*1000:.2f} ms")
    print(f"Std Dev: {std_dev*1000:.2f} ms")

if __name__ == "__main__":
    try:
        asyncio.run(run_benchmark())
    except KeyboardInterrupt:
        pass
