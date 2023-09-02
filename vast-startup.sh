apt-get update && apt-get upgrade
git config --global user.name [INSERT NAME]
git config --global user.email [INSERT EMAIL]
git config --global github.user [INSERT USERNAME]
git config --global github.token [INSERT TOKEN]
git clone https://github.com/Aaquib111/acdcpp.git
cd acdcpp
apt-get install graphviz-dev
pip install git+https://github.com/ArthurConmy/Automatic-Circuit-Discovery.git
pip install -r requirements.txt