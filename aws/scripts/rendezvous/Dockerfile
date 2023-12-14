FROM public.ecr.aws/docker/library/node:hydrogen-alpine3.19
RUN mkdir -p /rendezvous
WORKDIR /rendezvous
COPY . .

RUN npm run build

RUN npm install
# EXPOSE 3000
CMD [ "node", "server.js" ]