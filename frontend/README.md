# QueryHub Frontend

## Overview
A modern, responsive chat interface for QueryHub built with React, TypeScript, and Vite. Features real-time database connectivity, session management, and an intuitive user experience for querying multiple database systems.

🌐 **Live Application**: [example.com](https://example.com)

---

## ✨ Features

- **Real-time Chat Interface**: Interactive chat UI for natural language database queries
- **Multi-Database Support**: Connect and manage multiple database instances
- **Session Management**: Persistent chat sessions with history
- **Authentication**: Secure sign-in/sign-up flows
- **Protected Routes**: Role-based access control
- **Responsive Design**: Mobile-first design using Bootstrap
- **Type Safety**: Full TypeScript implementation
- **Docker Ready**: Containerized deployment with Nginx

---

## 🛠 Tech Stack

- **Framework**: [React 18](https://react.dev/) with [TypeScript](https://www.typescriptlang.org/)
- **Build Tool**: [Vite](https://vitejs.dev/)
- **UI Library**: [Bootstrap 5](https://getbootstrap.com/)
- **Routing**: [React Router](https://reactrouter.com/)
- **HTTP Client**: [Axios](https://axios-http.com/)
- **Web Server**: [Nginx](https://nginx.org/) (production)
- **Container**: [Docker](https://www.docker.com/)

---

## 📦 Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js**: v18.x or higher
- **npm**: v9.x or higher
- **Docker**: v20.x or higher (for containerized deployment)
- **Docker Compose**: v2.x or higher

---

## 🚀 Getting Started

### Environment Setup

1. **Clone the repository**
```bash
   git clone https://github.com/Northeastern-MLOps-Group-12/QueryHub.git
   cd QueryHub/frontend
```

2. **Create environment file**
   
   Create a `.env` file in the `frontend` directory:
```bash
   VITE_BACKEND_URL=http://localhost:8000
```
   
   Replace `http://localhost:8000` with your backend API URL.

### Local Development

1. **Install dependencies**
```bash
   npm install
```

2. **Start development server**
```bash
   npm run dev
```
   
   The application will be available at `http://localhost:5173`

### Docker Deployment

**Using Docker Compose** (recommended):
```bash
docker compose up --build
```

---

## 📁 Project Structure
```
frontend/
├── public/                               # Static assets
│   └── logo.png                          # Application logo
│
├── src/
│   ├── account/                          # Authentication pages
│   │   ├── index.tsx                     # Account routing wrapper
│   │   ├── SignIn.tsx                    # Sign-in page component
│   │   └── SignUp.tsx                    # Sign-up page component
│   │
│   ├── assets/                           # Images and static resources
│   │   └── default-avatar.png            # Default user avatar
│   │
│   ├── chat-interface/                   # Chat UI components
│   │   ├── index.tsx                     # Main chat interface container
│   │   ├── NewChatModal.css              # Modal styling
│   │   └── NewChatModal.tsx              # New chat session modal
│   │
│   ├── components/                       # Reusable components
│   │   └── ProtectedRoute.tsx            # Route authentication wrapper
│   │
│   ├── data/                             # Static data and configurations
│   │   ├── dpOptions.tsx                 # Database provider options
│   │   └── homeContent.tsx               # Home page content data
│   │
│   ├── database/                         # Database management features
│   │   ├── ConnectedDatabases.tsx        # List of connected databases
│   │   ├── DatabaseConnection.tsx        # Database connection form
│   │   ├── DatabaseDescription.tsx       # Database details view
│   │   ├── DatabaseEditor.tsx            # Edit database configurations
│   │   └── index.tsx                     # Database module routing
│   │
│   ├── home/                             # Landing page
│   │   └── index.tsx                     # Home page component
│   │
│   ├── hooks/                            # Custom React hooks
│   │   └── useAuth.tsx                   # Authentication state hook
│   │
│   ├── services/                         # API service layers
│   │   ├── authService.tsx               # Authentication API calls
│   │   ├── chatService.tsx               # Chat service calls
│   │   └── databaseService.tsx           # Database API calls
│   │
│   ├── App.css                           # Global application styles
│   ├── App.tsx                           # Root application component
│   ├── Footer.tsx                        # Footer component
│   ├── index.css                         # Global CSS reset and base styles
│   ├── main.tsx                          # Application entry point
│   └── Navbar.tsx                        # Navigation bar component
│
├── .dockerignore                         # Docker ignore patterns
├── .gitignore                            # Git ignore patterns
├── docker-compose.yml                    # Docker Compose configuration
├── Dockerfile                            # Multi-stage Docker build
├── eslint.config.js                      # ESLint configuration
├── index.html                            # HTML entry point
├── nginx.conf                            # Nginx server configuration
├── package-lock.json                     # Locked dependency versions
├── package.json                          # Project dependencies and scripts
├── README.md                             # This file
├── tsconfig.app.json                     # TypeScript app configuration
├── tsconfig.json                         # Base TypeScript configuration
├── tsconfig.node.json                    # TypeScript Node configuration
└── vite.config.ts                        # Vite build configuration
```

---

## 🚢 Deployment

### CI/CD Pipeline

The frontend is automatically deployed to **Google Cloud Run** using GitHub Actions.

**Trigger**: Push to `main` branch with changes in the `frontend/` directory

**Workflow**: `.github/workflows/frontend-deploy.yml`

**Deployment Steps**:
1. Checkout code
2. Authenticate to Google Cloud
3. Setup gcloud CLI
4. Configure Docker
5. Create frontend .env file
6. Build & Push Docker Image
7. Deploy to Cloud Run

### Environment Variables

Set the following environment variables in your Cloud Run service:
```bash
VITE_BACKEND_URL=https://your-backend-api.com
```