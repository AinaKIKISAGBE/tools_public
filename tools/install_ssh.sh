

sudo su 
apt-get update 




# apt-get update  
#>> apt-get install sudo -y
################### add user and set password  ####################
# RUN echo 'root:rootpw!' | chpasswd
# RUN echo root:root | chpasswd
#RUN echo 'root' | passwd --stdin root 
#RUN useradd -m root 
#>> echo "root:root" | chpasswd 
#>> adduser root sudo
#USER root
################### install ssh  ####################
#### docker : service  name commande
apt-get update
apt install net-tools
apt-get install ufw -y
apt install openssh-server -y
service ssh enable
ufw allow 22/tcp
ufw allow ssh  
ufw disable


# enlever le commentaire de #Port 22
sed -i 's/#Port 22/Port 22/' /etc/ssh/sshd_config
# enable root permission ssh remote
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
apt-get update
service ssh start 
service ssh status 

service ssh restart
service ssh reload
service ssh restart
#service ssh start 
#apt-get update 
#RUN service ssh start
