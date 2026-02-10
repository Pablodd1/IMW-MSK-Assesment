const http = require('http');

const BASE_URL = 'http://127.0.0.1:3000';
const ENDPOINTS = [
  '/',
  '/api/exercises',
  '/api/billing/codes'
  // /api/patients and others might require DB seeding or auth in future
];

async function checkEndpoint(path) {
  return new Promise((resolve, reject) => {
    const req = http.get(`${BASE_URL}${path}`, (res) => {
      // Allow redirects (3xx) and success (2xx)
      if (res.statusCode >= 200 && res.statusCode < 400) {
        console.log(`✅ ${path}: ${res.statusCode}`);
        resolve(true);
      } else {
        console.error(`❌ ${path}: ${res.statusCode}`);
        // Read body for error details
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
             if (data) console.error(`   Response: ${data.substring(0, 100)}...`);
             resolve(false);
        });
      }
    });

    req.on('error', (e) => {
      // Start server hint
      console.error(`❌ ${path}: Connection refused.`);
      console.error(`   Ensure the server is running: 'npm run dev' or 'npm run preview'`);
      resolve(false);
    });
  });
}

async function run() {
  console.log(`Testing API Endpoints at ${BASE_URL}...`);
  let success = true;
  let connectionFailed = false;

  for (const endpoint of ENDPOINTS) {
    const result = await checkEndpoint(endpoint);
    if (!result) {
        success = false;
        // If connection refused, stop trying others
        // We can't easily detect "ECONNREFUSED" from the result boolean here without changing checkEndpoint,
        // but typically all will fail if the server is down.
    }
  }

  if (!success) {
      console.log("\n⚠️  Some tests failed. If all failed, the server is likely not running.");
      process.exit(1);
  } else {
      console.log("\n✅ All endpoints reachable.");
  }
}

run();
