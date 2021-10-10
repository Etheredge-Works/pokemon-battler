#ARG WORKERS=16
FROM node
RUN git clone https://github.com/smogon/pokemon-showdown.git /pokemon-showdown
WORKDIR /pokemon-showdown
RUN npm install
RUN cp /pokemon-showdown/config/config-example.js /pokemon-showdown/config/config.js
RUN sed -i "s/exports.workers = 1;/exports.workers = 8;/g" /pokemon-showdown/config/config.js
RUN mkdir logs

CMD ["node", "pokemon-showdown", "start", "--no-security"]
