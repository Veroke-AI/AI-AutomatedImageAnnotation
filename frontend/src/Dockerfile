FROM node:18.19

# Set working directory inside container
WORKDIR /app

# Install Angular CLI globally
RUN npm install -g @angular/cli

# Copy Angular project files from context (which is ./gui)
COPY . /app

# Install dependencies
RUN npm install --legacy-peer-deps

# Expose Angular dev server port
EXPOSE 4200

# Start Angular dev server
CMD ["npm", "start"]
