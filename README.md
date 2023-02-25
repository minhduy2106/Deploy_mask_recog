# Deploy_mask_recog

docker build -t duylnm/mask_recog_web -f Dockerfile . 


docker run --name app -p 8080:8080 duylnm/mask_recog_web    
