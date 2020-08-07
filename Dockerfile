FROM ubuntu
MAINTAINER https://github.com/chineseocr/chineseocr
LABEL version="1.0"
EXPOSE 8080
RUN apt-get update
RUN apt-get install  libsm6 libxrender1 libxext-dev gcc -y
##下载Anaconda3 python 环境安装包 放置在chineseocr目录 url地址https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
WORKDIR /chineseocr
ADD . /chineseocr
RUN cd /chineseocr && sh -c '/bin/echo -e "\nyes\n\nyes" | sh Anaconda3-2019.03-Linux-x86_64.sh'
#RUN /root/anaconda3/bin/conda install -y python=3.6
RUN /root/anaconda3/bin/pip install easydict opencv-contrib-python==3.4.2.17 Cython h5py pandas requests bs4 matplotlib lxml -U pillow keras==2.3.1 tensorflow==1.15.3 -i https://pypi.doubanio.com/simple/
RUN /root/anaconda3/bin/pip install web.py==0.40.dev0
RUN /root/anaconda3/bin/conda install -y pytorch-cpu torchvision-cpu -c pytorch
RUN rm Anaconda3-2019.03-Linux-x86_64.sh
#RUN cd /chineseocr/text/detector/utils && sh make-for-cpu.sh
#RUN conda clean -p
#RUN conda clean -t
