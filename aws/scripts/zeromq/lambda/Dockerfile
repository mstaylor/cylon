FROM public.ecr.aws/lambda/python:3.8

# Install necessary build tools
RUN yum install -y gcc gcc-c++ make tar

# Download and compile ZeroMQ
RUN curl -sL https://github.com/zeromq/libzmq/releases/download/v4.3.2/zeromq-4.3.2.tar.gz | tar xz \
    && cd zeromq-4.3.2 \
    && ./configure --prefix=/var/task/lib --without-docs \
    && make \
    && make install

# Install pyzmq using the compiled ZeroMQ
RUN pip install pyzmq --no-binary :all: --install-option="--zmq=/var/task/lib"

# Copy compiled libraries to a folder that will be used as a layer
RUN mkdir /opt/python
RUN cp -r /var/task/lib/lib/* /opt/python/
RUN cp -r /var/lang/lib/python3.8/site-packages/zmq* /opt/python/

CMD ["bash"]
