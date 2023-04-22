
docker-compose -p=database_postgres up -d



docker-compose up -d 
docker-compose up -d --project-name database 

docker-compose -p project_name build
docker-compose --project-name foo build bar
docker-compose -p app up --build

docker-compose -f <docker_compose_file> -p "<project_name>" up
docker-compose -p="stack-example" up