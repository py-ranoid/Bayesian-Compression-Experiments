
sudo apt-get update
sudo apt-get --yes --force-yes install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update

sudo apt-get --yes --force-yes install docker-ce=17.12.0~ce-0~ubuntu

sudo docker version

echo "=====Add nvidia-docker package repositories========"
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

echo "Install nvidia-docker2 and reload the Docker daemon configuration"
sudo apt-get install -y nvidia-docker2=2.0.2+docker17.12.0-1 nvidia-container-runtime=1.1.1+docker17.12.0-1

sudo pkill -SIGHUP dockerd
sudo systemctl restart docker
sudo usermod -a -G docker $USER

sudo cat /etc/docker/daemon.json

echo "Set nvidia-docker as default runtime"
sudo sed -i '$s/}/,\n"default-runtime":"nvidia"}/' /etc/docker/daemon.json
sudo pkill -SIGHUP dockerd
sudo systemctl restart docker

sudo cat /etc/docker/daemon.json
