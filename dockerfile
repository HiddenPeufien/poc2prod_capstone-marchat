FROM python:3.7

WORKDIR /app

COPY . .

RUN conda env create -f environment.yml

RUN conda activate myenv
RUN echo "Make sure flask is installed:"

EXPOSE 5000

CMD ["python", "predict/predict/app.py" ]

