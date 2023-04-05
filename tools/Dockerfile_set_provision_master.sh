
### cmd1 && cmd2 : vérifie si la première commande s'est bien exécuter avant d'exécuter la suivante sinon le processus s'arrete 

### cmd1 & cmd2 : exécute les commandes sans vérifier si la commande précédente s'est bien exécuté ou pas



apt-get update  
apt-get install sudo -y



################### add user and set password  ####################
# RUN echo 'root:rootpw!' | chpasswd
# RUN echo root:root | chpasswd
#RUN echo 'root' | passwd --stdin root 
#RUN useradd -m root 
echo "root:root" | chpasswd 
adduser root sudo
#USER root

################### install ssh  ####################
#### docker : service  name commande
apt-get update
apt-get install ufw -y
apt install openssh-server -y
#service ssh enable
ufw allow 22/tcp
ufw allow ssh  
service ssh start 
#service ssh status 
# enlever le commentaire de #Port 22
sed -i 's/#Port 22/Port 22/' /etc/ssh/sshd_config
# enable root permission ssh remote
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
#service ssh restart
service ssh reload
service ssh start 
apt-get update 
#RUN service ssh start


################### remote docker  ####################
apt-get install docker.io -y


################### other  ####################
apt-get install psmisc -y # enable to run killall
apt-get install systemctl -y 

apt install python3-pip -y
apt install python3-venv -y
apt-get install curl -y

apt-get install openvpn unzip -y
apt install git -y




apt-get install xterm -y
#pip install psycopg2 or 

################### install package python  ####################
pip install psycopg2-binary numpy pandas 







mkdir /myapp/
chmod -R 777 /myapp 

################### clone tools_public  ####################
rm -r /myapp/tools_public 
mkdir /myapp/tools_public
git clone https://github.com/AinaKIKISAGBE/tools_public.git /myapp/tools_public

##### save github_key
python3 /myapp/tools_public/python_sql_database/save_github_key_on_vm.py

################### clone tools_private  ####################
GITHUB_TOKEN_keep=$(< /GITHUB_TOKEN)
rm -r /myapp/tools_private 
mkdir /myapp/tools_private
#git clone https://$GITHUB_TOKEN_keep@github.com/AinaKIKISAGBE/tools_private.git /myapp/tools_private or 
git clone https://${GITHUB_TOKEN_keep}@github.com/AinaKIKISAGBE/tools_private.git /myapp/tools_private
chmod -R 777 /myapp 






cp /myapp/tools_private/vm_provitionning/master/docker/Dockerfile_provision_master.sh  Dockerfile_provision_master.sh 
chmod +x Dockerfile_provision_master.sh 
/Dockerfile_provision_master.sh








