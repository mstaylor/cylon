const server = require('net').createServer(function (socket) {

});

server.listen(9999, function (err) {
    if (err) {
      return console.log(err);
    }

    console.log('server listening on', server.address().address + ':' + server.address().port);
});