sudo apt-get update
sudo apt-get -y install python3 python3-pip
sudo ln -s /bin/python3 /bin/python
pip3 install --no-cache-dir -r requirements_aws.txt

pthdir=models/pix2pix/checkpoints/pix2pix_vase_fragments_512/
mkdir -p $pthdir
sudo apt-get -y install awscli
sudo aws s3 cp s3://vasegen-matched-512/latest_net_G.pth $pthdir

sudo apt-get -y install apache2 libapache2-mod-wsgi

sudo ln -sT src/web /var/www/html/flaskapp


echo "Add the following to"
echo "/etc/apache2/sites-enabled/000-default.conf"
cat src/web/mod_wsgi.txt

echo
echo "Then call \"sudo service apache2 restart\""
