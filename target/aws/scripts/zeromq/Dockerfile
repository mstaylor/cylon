FROM public.ecr.aws/docker/library/node:hydrogen-alpine3.19
RUN mkdir -p /zeromqtest
WORKDIR /zeromqtest
COPY . .
RUN npm install
RUN npm run build

# EXPOSE 3000
CMD [ "node", "dist/server.js" ]