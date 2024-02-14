sudo docker build -f Dockerfile.annoy -t annoy_api ..
sudo docker build -f Dockerfile.webapp -t recommender_webapp ..
# sudo docker build --no-cache -f Dockerfile.annoy -t annoy_api ..
# sudo docker build --no-cache -f Dockerfile.webapp -t recommender_webapp ..
echo "ALL DONE"
