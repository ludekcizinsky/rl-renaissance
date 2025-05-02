# 1. Base image
FROM python:3.6-buster

# 2. System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2-dev \
    libxslt1-dev \
    libopenblas-dev \
    liblapack-dev \
    less \
    build-essential \
    gfortran \
    fort77 \
    wget \
    cmake \
    libflint-2.5.2 \
    libflint-dev \
    libgmp-dev \
    yasm \
    xvfb \
    xauth \
    ffmpeg \
    firefox-esr \
    && rm -rf /var/lib/apt/lists/*

# 3. Create non-root user
ENV USER=renaissance \
    HOME=/home/renaissance
RUN useradd -ms /bin/bash $USER

# 4. Switch to that user
USER $USER
WORKDIR $HOME

# 5. Copy utility scripts and make executable
USER root
COPY docker/utils/ /utils/
RUN chmod +x /utils/*.sh

# 6. Python tooling
RUN python3 -m pip install --upgrade pip pipenv

# 7. Install your Python requirements
COPY docker/requirements.txt .
RUN pip install -r requirements.txt

# 8. Build & install SUNDIALS
RUN /utils/install_sundials.sh

# 9. Point compilers at SUNDIALS
ENV SUNDIALS_INST=${HOME}/sundials-5.1.0
ENV SUNDIALS_INCLUDEDIR=${SUNDIALS_INST}/include
ENV SUNDIALS_LIBDIR=${SUNDIALS_INST}/lib
ENV LD_LIBRARY_PATH=${SUNDIALS_LIBDIR}:/usr/local/lib
ENV CFLAGS="-I${SUNDIALS_INCLUDEDIR} -fPIC"
ENV FFLAGS="-fPIC"
ENV LDFLAGS="-L${SUNDIALS_LIBDIR}"

# 10. Install the Python SUNDIALS interface
RUN pip install scikits.odes==2.6.3

# 11. CPLEX & Gurobi (optional)
COPY docker/solvers/ /solvers/
RUN /utils/install_cplex.sh \
    && /utils/install_gurobi.sh \
    && rm -rf /solvers \
    && /utils/activate_gurobi.sh

# 12. Bring in your .bashrc
COPY docker/.bashrc $HOME/.bashrc
RUN chown $USER:$USER $HOME/.bashrc

# 13. Copy in your Renaissance project (at repo root)
COPY . /home/renaissance/renaissance

# 14. Bake in your package (Cython extensions + [ORACLE] extras)
RUN pip install /home/renaissance/renaissance[ORACLE]

# 15. Final unprivileged user
USER $USER

# 16. Create and switch to a workspace
RUN mkdir -p $HOME/work
WORKDIR $HOME/work

# 17. Default command
CMD ["bash"]
