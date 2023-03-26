service_create=$1
link_service_create=$2

#service_create=worker_update_iptable
#link_service_create=/myapp/tools_private/vm_provitionning/worker/${service_create}.sh

killall ${service_create} 
systemctl stop ${service_create}.service 
rm /etc/systemd/system/multi-user.target.wants/${service_create}.service 
rm /usr/local/bin/${service_create}.sh 
touch /usr/local/bin/${service_create}.sh 
echo "${link_service_create}" >> /usr/local/bin/${service_create}.sh 
chmod 777 ${link_service_create}
rm /etc/systemd/system/${service_create}.service 
touch /etc/systemd/system/${service_create}.service 
echo "[Unit] " >> /etc/systemd/system/${service_create}.service  
echo "Description=Gateway activation for private network " >> /etc/systemd/system/${service_create}.service 
echo "After=network-online.target " >> /etc/systemd/system/${service_create}.service 
echo "[Service] " >> /etc/systemd/system/${service_create}.service 
echo "Type=oneshot " >> /etc/systemd/system/${service_create}.service 
echo "RemainAfterExit=yes " >> /etc/systemd/system/${service_create}.service 
echo "ExecStart=/bin/bash /usr/local/bin/${service_create}.sh start  " >> /etc/systemd/system/${service_create}.service 
echo "[Install] " >> /etc/systemd/system/${service_create}.service 
echo "WantedBy=multi-user.target" >> /etc/systemd/system/${service_create}.service 
chmod 777 /etc/systemd/system/${service_create}.service 
rm /etc/systemd/system/multi-user.target.wants/${service_create}.service 
ln -s /lib/systemd/system/${service_create}.service /etc/systemd/system/multi-user.target.wants/ 
systemctl daemon-reload 
systemctl enable ${service_create}.service  
systemctl start ${service_create}.service