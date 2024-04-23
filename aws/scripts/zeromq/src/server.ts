import * as zmq from 'zeromq';

async function run(): Promise<void> {
  const sock = new zmq.Publisher;

  await sock.bind('tcp://*:5555');
  console.log('Publisher bound to port 5555');

  while (true) {
    await sock.send('Hello from ZeroMQ!');
    console.log('Sent: Hello from ZeroMQ!');
    await new Promise(resolve => setTimeout(resolve, 5000)); // wait for 5 seconds
  }
}

run().catch(err => console.error(err));
