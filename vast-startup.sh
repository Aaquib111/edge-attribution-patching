github_username=# INSERT USERNAME HERE
github_token=#INSERT TOKEN HERE
email=# INSERT EMAIL HERE

apt update && apt upgrade

git init
git remote set-url origin https://Aaquib111:$github_token@github.com//acdcpp.git
git config --global user.name $github_username
git config --global user.email $email
git config --global github.user $github_username 
git config --global github.token $github_token

apt install graphviz
apt install graphviz-dev

cd ..
git clone https://github.com/Aaquib111/Automatic-Circuit-Discovery.git
cd acdcpp/Automatic-Circuit-Discovery/
git submodule init
git submodule update
git pull origin master

pip install plotly transformer_lens pygraphviz
pip install -r requirements.txt
