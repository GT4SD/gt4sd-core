# intermediate image for dependencies and user "gt4sd" setup
FROM drugilsberg/gt4sd-base:main
ENV PATH=/opt/conda/envs/gt4sd/bin/:${PATH}
COPY notebooks/requirements.txt notebooks_requirements.txt
RUN pip install --no-cache-dir notebook==5.* tensorflow-cpu==2.1.0 gt4sd>=0.31.0 && pip install --no-cache-dir -r notebooks_requirements.txt
RUN adduser --disabled-password --gecos '' gt4sd
ENV HOME /home/gt4sd
COPY notebooks/ ${HOME}/notebooks/
RUN chown -R gt4sd:gt4sd ${HOME}
# run jupyter
USER gt4sd
WORKDIR ${HOME}
EXPOSE 8888
ENTRYPOINT []
CMD [ "jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]
