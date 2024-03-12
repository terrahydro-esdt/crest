FROM continuumio/miniconda3:latest

SHELL ["/bin/bash", "-c"]

RUN conda install -n base conda-libmamba-solver -q
RUN conda config --set solver libmamba

WORKDIR /crest/

COPY cicd/ /crest/cicd/.
COPY cicd/ /crest/docs/.
COPY crest/ /crest/crest/.
COPY tests/ /crest/tests/.
COPY examples/ /crest/examples/.

RUN conda install pytest
RUN conda env create -f cicd/environment_cpu.yaml -q

# Make RUN commands use the new environment:
SHELL ["conda", "run", "--no-capture-output", "-n", "crest_cpu", "/bin/bash", "-c"]

RUN source activate crest_cpu
RUN pytest --color=yes tests