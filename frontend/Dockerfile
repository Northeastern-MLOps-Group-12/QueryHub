# ----------------------------
# Stage 1: Build React/Vite App
# ----------------------------
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci --silent

# Copy source code and build
COPY . .
RUN npm run build

# ----------------------------
# Stage 2: Serve via Nginx
# ----------------------------
FROM nginx:alpine

# Copy built files from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom Nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80 for Cloud Run
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]