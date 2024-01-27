const server = require('net').createServer(function (socket) {
    Connects(socket);
});

var details = {
    name: 'No Name',
    localAddress: null,
    localPort: null,
    remoteAddress: null,
    remotePort: null
};

const port = process.env.LISTEN_PORT ?? "9999"

server.listen(port, function (err) {
    if (err) {
      return console.log(err);
    }

    console.log('server listening on', server.address().address + ':' + server.address().port);
});

function Connects (socket) {

    console.log('> (A) assuming A is connecting');
    console.log('> (A) remote address and port are:', socket.remoteAddress, socket.remotePort);
    console.log('> (A) storing this for when B connects');



    const array = socket.remoteAddress.split(':')
    const remoteIP = array[array.length - 1]

    details.remoteAddress = remoteIP;
    

    details.remotePort = socket.remotePort;

    socket.on('data', function (data) {
        console.log('> (A) incomming data from A:', data.toString());
        try {
            let localDataA = JSON.parse(data.toString());

            const address = socket.handshake.headers["x-forwarded-for"].split(",")[0];
            console.log(address);

            console.log('> (A) storing this for when B connects');
            console.log('');
            details.localAddress = localDataA.localAddress;
            details.localPort = localDataA.localPort;
            console.log('> (A) sending remote details back to A');
            socket.write(JSON.stringify(details));

            console.log('> (A)', address.address + ':' + address.port, '===> (NAT of A)',
                details.remoteAddress + ':' + details.remotePort, '===> (S)', socket.localAddress + ':' + socket.localPort);
        } catch (e) {
            console.log("exception ", e)
        }
    });

    socket.on('end', function () {
        console.log('> socket connection closed.');

    });

    socket.on('error', function (err) {
        console.log('> socket connection closed with err (',err,').');

    });
}