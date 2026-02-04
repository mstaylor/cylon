#!/usr/bin/env node
/**
 * Distributed WASM Test Runner
 *
 * Spawns multiple worker processes to test distributed operations through FMI/Redis.
 *
 * Usage: node distributed-test.js [worldSize] [redisHost] [redisPort]
 */

const { spawn } = require('child_process');
const path = require('path');

const worldSize = parseInt(process.argv[2] || '2', 10);
const redisHost = process.argv[3] || process.env.REDIS_HOST || 'localhost';
const redisPort = process.argv[4] || process.env.REDIS_PORT || '6379';
const commName = `cylon_dist_test_${Date.now()}`;

console.log('='.repeat(60));
console.log('Distributed WASM Test');
console.log('='.repeat(60));
console.log(`World size: ${worldSize}`);
console.log(`Redis: ${redisHost}:${redisPort}`);
console.log(`Comm name: ${commName}`);
console.log('='.repeat(60));

const workerScript = path.join(__dirname, 'distributed-worker.js');
const workers = [];
const results = new Map();

// Spawn workers
for (let rank = 0; rank < worldSize; rank++) {
  console.log(`Spawning worker ${rank}...`);

  const worker = spawn('node', [
    workerScript,
    rank.toString(),
    worldSize.toString(),
    commName,
    redisHost,
    redisPort,
  ], {
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  workers.push({ rank, process: worker });

  // Collect stdout
  worker.stdout.on('data', (data) => {
    const lines = data.toString().trim().split('\n');
    lines.forEach(line => console.log(line));
  });

  // Collect stderr
  worker.stderr.on('data', (data) => {
    const lines = data.toString().trim().split('\n');
    lines.forEach(line => console.error(line));
  });

  // Handle exit
  worker.on('close', (code) => {
    results.set(rank, code);
    console.log(`Worker ${rank} exited with code ${code}`);

    // Check if all workers have finished
    if (results.size === worldSize) {
      const allPassed = Array.from(results.values()).every(c => c === 0);
      console.log('');
      console.log('='.repeat(60));
      if (allPassed) {
        console.log('ALL WORKERS PASSED');
        console.log('='.repeat(60));
        process.exit(0);
      } else {
        console.log('SOME WORKERS FAILED');
        results.forEach((code, rank) => {
          console.log(`  Worker ${rank}: ${code === 0 ? 'PASSED' : 'FAILED'}`);
        });
        console.log('='.repeat(60));
        process.exit(1);
      }
    }
  });
}

// Timeout after 60 seconds
setTimeout(() => {
  console.error('TIMEOUT: Test did not complete within 60 seconds');
  workers.forEach(({ process }) => process.kill('SIGTERM'));
  process.exit(1);
}, 60000);