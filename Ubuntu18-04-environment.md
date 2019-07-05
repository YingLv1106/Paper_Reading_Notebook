---
title: Ubuntu18.04-environment
tags:
---
-one  
look giuhub [[lvying]](https://github.com/cocodataset/cocoapi)
#### 1.install Ubuntu18.04 and update sourc
Update to mirrors.aliyun.com, reference [this](https://blog.csdn.net/hymanjack/article/details/80285400) in step 1.
#### 2.Deep learning environment
Anaconda3.5.2-py36+cuda10.0+cudnn7.5+pytorch1.1+tensorflow2.0-beta+mxnet
##### Anaconda
- download [Anaconda3-5.2.0-Linux-x86_64](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.2.0-Linux-x86_64.sh) or [other version](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/) 
- `bash Anaconda3-5.2.0-Linux-x86_64.sh ` 
- enter `yes` for adding path to .bashrc, `no` for install vscode
- `source ~/.bashrc`
- use `which pip` or `which python`, you will see default pip or python is now in Anaconda.
- use `conda -V` to version of Anaconda
- add conda mirror follow [Anaconda 镜像使用帮助](https://mirror.tuna.tsinghua.edu.cn/help/anaconda/)
- update pip source, `mkdir ~/.pip` && `sudo gedit ~/.pip/pip.conf `, then add the following in pip.conf
    ```
        [global]
        trusted-host=mirrors.aliyun.com
        index-url=https://mirrors.aliyun.com/pypi/simple/
    ```
##### NVIDIA driver
- `sudo apt purge nvidia-*` to remove old driver
- `sudo add-apt-repository ppa:graphics-drivers/ppa` to add repository
- `sudo apt update && sudo apt upgrade`
- `ubuntu-drivers devices`  to see driver versions
- `sudo apt install nvidia-driver-410` to install
- restart and check install `nvidia-smi` or `watch -n 0.1 nvidia-smi`
- use `sudo apt-mark hold nvidia-driver-410` to prevent graphics drivers from updating automatically

##### CUDA 
- download [cuda_10.0.130_410.48_linux.run](https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux) or search for [other version](https://developer.nvidia.com/cuda-zone)
- `sudo sh cuda_10.0.130_410.48_linux.run` 
- use `q` to install directly and pay attention not to install NVIDIA Accelerated Graphics Driver, the other's can use `y` or default. Details to see [reference](https://blog.csdn.net/qq_32408773/article/details/84112166)
- add following use `sudo gedit ~/.bashrc`
    ```
    export CUDA_HOME=/usr/local/cuda 
    export PATH=$PATH:$CUDA_HOME/bin 
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    ```
- `source ~/.bashrc` and use `nvcc -V` to see cuda version

##### cudnn
- download [cudnn-10.0-linux-x64-v7.5.0.56.solitairetheme8](https://developer.download.nvidia.cn/compute/machine-learning/cudnn/secure/v7.5.0.56/prod/10.0_20190219/cudnn-10.0-linux-x64-v7.5.0.56.tgz?rIGNHCi7ouqaocMZS_7MIYDmirXC-NaXZ6O05M1wFptPs6erdFe6PdoRf33k6jIbKMbf8-IhhfkEbBinatU9Jt7d-HBC3ru4VfUTk2tV4_Y1bMRYNbmfwqH_pL0awE_DNb6M7tt164WR15RpB-3YqwcE3AgF04L7Vc0tGpiJAhaTiKYr6RTFP3U3rrfSio4hVOoXIXHmamSxprR6pYptIrGpsLM) or [other version](https://developer.nvidia.com/rdp/cudnn-archive)
- then
    ```
    tar -zxvf cudnn-10.0-linux-x64-v7.5.0.56.solitairetheme8
    sudo cp cuda/include/cudnn.h /usr/local/cuda/include/ 
    sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/ 
    sudo chmod a+r /usr/local/cuda/include/cudnn.h 
    sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
    cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
    ```

##### pytorch
- see https://pytorch.org/get-started/locally/
- or use downloaded .whl files 
    ```
    pip install torch-1.1.0-cp36-cp36m-linux_x86_64.whl
    pip install torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
    ```
- test install
    ``` python
    In [1]: import torch
    In [2]: torch.__version__
    Out[2]: '1.1.0'
    In [3]: torch.cuda.is_available()
    Out[3]: True
    ```

##### tensorflow2.0-beta
- install use pip
    ```
    pip install tensorflow==2.0.0-beta0  # cpu
    pip install tensorflow-gpu==2.0.0-beta0  # gpu
    ```
- It may have problem `ERROR: Cannot uninstall 'wrapt'`. Directly use  global search to remove related folders and `.Egg-info` related files. See [this](https://www.jianshu.com/p/28e2ae6fbd75) for more details.
- test install 
    ```
    In [1]: import tensorflow as tf
    In [2]: tf.__version__
    Out[2]: '2.0.0-beta0'
    In [3]: tf.test.is_gpu_available()
    Out[3]: True
    ```

##### mxnet
```
pip install mxnet-cu100
```

#### Applications(some are in [reference](https://blog.csdn.net/hymanjack/article/details/80285400))

##### sougou input
- download [sougou for Linux](https://pinyin.sogou.com/linux/?r=pinyin)
- see https://blog.csdn.net/lupengCSDN/article/details/80279177

##### WPS
- download [WPS](https://www.v2ex.com/t/460732) and install directly

##### minmize when clik
```
gsettings set org.gnome.shell.extensions.dash-to-dock click-action 'minimize'
```

##### vim & Compression software & support exfat
```
sudo apt install vim
sudo apt install p7zip-full p7zip-rar rar unzip
sudo apt install exfat-fuse exfat-utils
```

##### git
- add your .ssh to your home and init
    ```
    sudo apt-get install git
    git config --global user.name ''
    git config --global user.email 
    ```

##### opencv for python
```
sudo apt install libopencv-dev
pip install opencv-python
```

##### pycharm
- install in Ubuntu Software
- or [this reference](https://blog.csdn.net/WANG_yu09/article/details/81630941), download [pycharm Community version](https://www.jetbrains.com/pycharm/download/#section=linux)
- `tar -zxvf pycharm-community-{your version}.tar.gz`
- `sudo mv pycharm-community-{your version} /opt/`
- `sudo gedit /usr/share/applications/pycharm.desktop` and add the follows
    ```
    [Desktop Entry]
    Version=1.0
    Type=Application
    Name=Pycharm
    Icon=/opt/pycharm-community-2019.1.3/bin/pycharm.png
    Exec=/opt/pycharm-community-2019.1.3/bin/pycharm.sh
    MimeType=application/x-py;
    Name[en_US]=pycharm
    ```
- `sudo chmod u+x /usr/share/applications/pycharm.desktop`
- It may have problem `No module named 'distutils.core'`, solve it by `sudo apt-get install python3-distutils`



#### 3.use google

##### install shadowsocks
- install python-pip
    ```
    sudo apt-get update
    sudo apt-get install python-pip
    ```
- install version 3.0.0
    ```
    sudo pip install -U git+https://github.com/shadowsocks/shadowsocks.git@master
    ```
- edit .json config file `sudo mkdir /etc/shadowsocks` && `sudo gedit /etc/shadowsocks/shadowsocks.json` and add follows
    ```json
    {
    "server":"server_ip",
    "server_port":server_port,
    "local_address":"127.0.0.1",
    "local_port":1080,
    "password":"mypassword",
    "timeout":300,
    "method":"aes-256-cfb"
    }
    ```
- turn on/off
    ```
    sslocal -c /etc/shadowsocks/shadowsocks.json -d  start
    sslocal -c /etc/shadowsocks/shadowsocks.json -d  stop
    ```

- easy trun on/off (for [reference](https://blog.csdn.net/qq_24406903/article/details/85011090)), create file `sudo gedit ssll` and add

    ```
    #!/bin/bash
    if [ "$1" == "on" ]
    then
        sslocal -c /etc/shadowsocks/shadowsocks.json -d  start
    elif [ "$1" == "off" ]
    then
        sslocal -c /etc/shadowsocks/shadowsocks.json -d stop
    fi
    ```
    save and execute the commands
    ```
    chmod +x ssll
    sudo mv ssll /bin/

    # use
    sudo ssll on    # turn on
    sudo ssll off    # turn off
    ```


##### if you use shadowsocks-qt5

```
sudo add-apt-repository ppa:hzwhuang/ss-qt5
sudo gedit /etc/apt/sources.list.d/hzwhuang-ubuntu-ss-qt5-bionic.list 
# then change bionic(for 18.04) to xenial(for 16.04)
sudo apt-get update
sudo apt-get install shadowsocks-qt5
```

##### PAC
- install
    ```
    sudo pip install genpac
    sudo pip install --upgrade genpac
    ```
- generaye autoproxy.pac
    ```
    sudo genpac --proxy="SOCKS5 127.0.0.1:1080" -o autoproxy.pac --gfwlist-url="https://raw.githubusercontent.com/gfwlist/gfwlist/master/gfwlist.txt"
    ```

- Settings -> Network -> Network Proxy -> choose Automatic 
  -> add path `file:// + {path of autoproxy.pac}`


##### startup
Startup Application add rule name `shadowsocks`, rule command 
`sudo /usr/local/bin/sslocal -c /etc/shadowsocks/shadowsocks.json -d start`

##### chrome (follow [this reference](http://www.modys.top/ubuntu-configuration-google-browser-shadowsocks-proxy-online.html))



