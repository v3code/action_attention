FROM python:3.7
WORKDIR /code
COPY ./requirements.txt ./
COPY ./init.sh .
RUN ./init.sh
COPY . .
ENTRYPOINT ["tail", "-f", "/dev/null"]