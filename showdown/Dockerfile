#ARG WORKERS=16
FROM node:latest
RUN git clone -b v0.11.7 https://github.com/smogon/pokemon-showdown.git /pokemon-showdown
WORKDIR /pokemon-showdown
RUN npm install 
RUN cp /pokemon-showdown/config/config-example.js /pokemon-showdown/config/config.js
# See showdown config for 100s of clients comment 
RUN sed -i "s/exports.workers = 1;/exports.workers = 8;/g" /pokemon-showdown/config/config.js
RUN sed -i "s/exports.validatorprocesses = 1;/exports.validatorprocesses = 4;/g" /pokemon-showdown/config/config.js
RUN sed -i "s/exports.simulatorprocesses = 1;/exports.simulatorprocesses = 8;/g" /pokemon-showdown/config/config.js
RUN mkdir -p logs

CMD ["node", "pokemon-showdown", "start", "--no-security"]
# CMD ["node", "pokemon-showdown"]
