
GITHUB_TOKEN_keep=$(< /GITHUB_TOKEN)
rm -r /myapp/tools_private & mkdir /myapp/tools_private
#git clone https://$GITHUB_TOKEN_keep@github.com/AinaKIKISAGBE/tools_private.git /myapp/tools_private or 
git clone https://${GITHUB_TOKEN_keep}@github.com/AinaKIKISAGBE/tools_private.git /myapp/tools_private
chmod -R 777 /myapp 
