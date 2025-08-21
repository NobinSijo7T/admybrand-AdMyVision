# AdMySense - Docker Deployment Guide

## Quick Start

### Option 1: Using Docker Compose (Recommended)
```bash
# Clone the repository
git clone https://github.com/NobinSijo7T/admybrand-AdMyVision.git
cd admybrand-AdMyVision

# Build and run with Docker Compose
docker-compose up --build
```

### Option 2: Using Docker directly
```bash
# Build the Docker image
docker build -t admysense .

# Run the container
docker run -p 8501:8501 admysense
```

### Option 3: Using run scripts
```bash
# Windows
run.bat

# Linux/macOS
./run.sh
```

## Accessing the Application

Once running, open your browser and navigate to:
- **Local access**: http://localhost:8501
- **Network access**: http://YOUR_IP_ADDRESS:8501

## Docker Commands

### Build the image
```bash
docker build -t admysense .
```

### Run the container
```bash
docker run -d -p 8501:8501 --name admysense-app admysense
```

### Stop the container
```bash
docker stop admysense-app
```

### Remove the container
```bash
docker rm admysense-app
```

### View logs
```bash
docker logs admysense-app
```

### Access container shell
```bash
docker exec -it admysense-app /bin/bash
```

## Docker Compose Commands

### Start services
```bash
docker-compose up -d
```

### Stop services
```bash
docker-compose down
```

### View logs
```bash
docker-compose logs -f
```

### Rebuild and restart
```bash
docker-compose up --build
```

## Environment Variables

You can customize the deployment using environment variables:

```bash
docker run -p 8501:8501 \
  -e STREAMLIT_SERVER_PORT=8501 \
  -e STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
  admysense
```

## Troubleshooting

### Port already in use
If port 8501 is already in use, change the port mapping:
```bash
docker run -p 8502:8501 admysense
```
Then access via http://localhost:8502

### Permission issues
If you encounter permission issues, try running with sudo:
```bash
sudo docker-compose up --build
```

### Container won't start
Check the logs for error messages:
```bash
docker logs admysense-app
```

## Production Deployment

For production deployment, consider:

1. **Using a reverse proxy** (nginx, Apache)
2. **Setting up SSL/TLS certificates**
3. **Configuring proper resource limits**
4. **Setting up monitoring and logging**

Example with resource limits:
```bash
docker run -d \
  -p 8501:8501 \
  --memory="2g" \
  --cpus="1.0" \
  --restart=unless-stopped \
  --name admysense-app \
  admysense
```

## Features

- ✅ Real-time object detection
- ✅ PC and mobile camera support
- ✅ Voice feedback system
- ✅ Modern responsive UI
- ✅ Cross-platform compatibility
- ✅ Docker containerization
- ✅ Health checks included
