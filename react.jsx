import React, { useState } from 'react';
import { Database, Cloud, Activity, Bell, GitBranch, Container, BarChart3, Cpu, RefreshCw, TrendingUp } from 'lucide-react';

const ArchitectureDiagram = () => {
  const [activeComponent, setActiveComponent] = useState(null);

  const components = {
    dataIngestion: {
      title: "Data Ingestion Layer",
      description: "Fetches real-time and historical crypto data from exchanges",
      tech: "ccxt library, Binance/Coinbase API",
      details: [
        "Pulls OHLCV data every 5-15 minutes",
        "Handles rate limiting and errors",
        "Stores raw data in PostgreSQL",
        "Archives historical data in Parquet"
      ]
    },
    featureEngine: {
      title: "Feature Engineering",
      description: "Calculates volatility metrics and technical indicators",
      tech: "Pandas, NumPy",
      details: [
        "Compute log returns",
        "Calculate realized volatility (rolling std)",
        "Generate technical indicators (RSI, MACD)",
        "Create lagged features for time series"
      ]
    },
    training: {
      title: "Model Training Pipeline",
      description: "Trains and evaluates GARCH and LSTM models",
      tech: "arch library, TensorFlow/PyTorch, Scikit-learn",
      details: [
        "GARCH(1,1) for traditional approach",
        "LSTM with 2-3 layers for deep learning",
        "Walk-forward validation",
        "Model comparison and selection",
        "Hyperparameter tuning with Optuna"
      ]
    },
    mlflow: {
      title: "Experiment Tracking",
      description: "Tracks experiments, metrics, and model versions",
      tech: "MLflow, Weights & Biases",
      details: [
        "Log hyperparameters and metrics",
        "Version control for models",
        "Compare model performance",
        "Store model artifacts"
      ]
    },
    cicd: {
      title: "CI/CD Pipeline",
      description: "Automated testing, training, and deployment",
      tech: "GitHub Actions",
      details: [
        "Scheduled weekly retraining",
        "Run unit tests on new code",
        "Build and push Docker images",
        "Deploy to cloud if metrics improve",
        "Rollback on failure"
      ]
    },
    docker: {
      title: "Containerization",
      description: "Packages application for consistent deployment",
      tech: "Docker, Docker Compose",
      details: [
        "Multi-stage builds for optimization",
        "Separate containers for API, training, monitoring",
        "Environment-specific configurations",
        "Image versioning with tags"
      ]
    },
    api: {
      title: "Prediction API",
      description: "Serves volatility predictions via REST API",
      tech: "FastAPI, Uvicorn",
      details: [
        "POST /predict endpoint",
        "Input: recent OHLCV data",
        "Output: 24h volatility forecast + confidence",
        "Sub-100ms latency requirement",
        "Swagger/OpenAPI documentation"
      ]
    },
    deployment: {
      title: "Cloud Deployment",
      description: "Hosts API and services in the cloud",
      tech: "AWS ECS / GCP Cloud Run / Azure Container Instances",
      details: [
        "Auto-scaling based on traffic",
        "Load balancing",
        "SSL/TLS encryption",
        "Environment variables for secrets",
        "Health check endpoints"
      ]
    },
    monitoring: {
      title: "Model Monitoring",
      description: "Tracks model performance and data drift",
      tech: "Prometheus, Grafana, Custom dashboards",
      details: [
        "Track prediction error over time",
        "Detect distribution shifts",
        "Monitor API latency and uptime",
        "Alert on performance degradation",
        "Log all predictions for analysis"
      ]
    },
    alerts: {
      title: "Alert System",
      description: "Notifies users when high volatility is predicted",
      tech: "Twilio, SendGrid, Slack API",
      details: [
        "Configurable volatility thresholds",
        "Multi-channel alerts (email, SMS, Slack)",
        "Alert history and analytics",
        "Deduplication to prevent spam",
        "Risk level classification (Low/Medium/High)"
      ]
    },
    dashboard: {
      title: "Interactive Dashboard",
      description: "Visualizes predictions and model performance",
      tech: "Plotly Dash, Streamlit",
      details: [
        "Real-time volatility charts",
        "Predicted vs actual comparison",
        "Model performance metrics",
        "Feature importance plots",
        "Historical alert log"
      ]
    },
    database: {
      title: "Data Storage",
      description: "Stores market data, predictions, and metrics",
      tech: "PostgreSQL, Redis (cache)",
      details: [
        "Time-series optimized tables",
        "Indexed for fast queries",
        "Redis for caching recent predictions",
        "Automated backups",
        "Data retention policies"
      ]
    }
  };

  const ComponentCard = ({ id, icon: Icon, title, color }) => (
    <div
      onClick={() => setActiveComponent(activeComponent === id ? null : id)}
      className={`cursor-pointer p-4 rounded-lg border-2 transition-all ${
        activeComponent === id
          ? `border-${color}-500 bg-${color}-50 scale-105`
          : 'border-gray-300 bg-white hover:border-gray-400'
      }`}
    >
      <div className="flex items-center gap-2 mb-2">
        <Icon className={`text-${color}-600`} size={24} />
        <h3 className="font-semibold text-sm">{title}</h3>
      </div>
      <p className="text-xs text-gray-600">{components[id].tech}</p>
    </div>
  );

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50">
      <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">
          Cryptocurrency Volatility Forecasting System
        </h1>
        <p className="text-gray-600 mb-4">
          End-to-End MLOps Pipeline for Bitcoin/Ethereum Volatility Prediction
        </p>
        <div className="flex gap-2 text-sm">
          <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full">Time Series</span>
          <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full">MLOps</span>
          <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full">FinTech</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* Left Column - Data Pipeline */}
        <div className="space-y-4">
          <h2 className="text-xl font-bold text-gray-700 mb-3 flex items-center gap-2">
            <Database size={24} className="text-blue-600" />
            Data Pipeline
          </h2>
          <ComponentCard id="dataIngestion" icon={TrendingUp} title="Data Ingestion" color="blue" />
          <ComponentCard id="featureEngine" icon={Cpu} title="Feature Engineering" color="blue" />
          <ComponentCard id="database" icon={Database} title="Data Storage" color="blue" />
        </div>

        {/* Middle Column - ML Pipeline */}
        <div className="space-y-4">
          <h2 className="text-xl font-bold text-gray-700 mb-3 flex items-center gap-2">
            <Activity size={24} className="text-green-600" />
            ML Pipeline
          </h2>
          <ComponentCard id="training" icon={RefreshCw} title="Model Training" color="green" />
          <ComponentCard id="mlflow" icon={BarChart3} title="Experiment Tracking" color="green" />
          <ComponentCard id="cicd" icon={GitBranch} title="CI/CD Pipeline" color="green" />
          <ComponentCard id="docker" icon={Container} title="Containerization" color="green" />
        </div>

        {/* Right Column - Deployment & Monitoring */}
        <div className="space-y-4">
          <h2 className="text-xl font-bold text-gray-700 mb-3 flex items-center gap-2">
            <Cloud size={24} className="text-purple-600" />
            Deployment
          </h2>
          <ComponentCard id="api" icon={Activity} title="Prediction API" color="purple" />
          <ComponentCard id="deployment" icon={Cloud} title="Cloud Hosting" color="purple" />
          <ComponentCard id="monitoring" icon={Activity} title="Monitoring" color="purple" />
          <ComponentCard id="alerts" icon={Bell} title="Alert System" color="purple" />
          <ComponentCard id="dashboard" icon={BarChart3} title="Dashboard" color="purple" />
        </div>
      </div>

      {/* Details Panel */}
      {activeComponent && (
        <div className="bg-white rounded-xl shadow-lg p-6 border-2 border-indigo-300">
          <h2 className="text-2xl font-bold text-gray-800 mb-2">
            {components[activeComponent].title}
          </h2>
          <p className="text-gray-600 mb-4">
            {components[activeComponent].description}
          </p>
          <div className="mb-4">
            <span className="font-semibold text-gray-700">Tech Stack: </span>
            <span className="text-indigo-600">{components[activeComponent].tech}</span>
          </div>
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">Key Features:</h3>
            <ul className="space-y-2">
              {components[activeComponent].details.map((detail, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <span className="text-indigo-600 mt-1">●</span>
                  <span className="text-gray-700">{detail}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Data Flow Diagram */}
      <div className="mt-6 bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-bold text-gray-800 mb-4">Data Flow</h2>
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-center">
          <div className="flex-1">
            <div className="bg-blue-100 rounded-lg p-4 mb-2">
              <TrendingUp className="mx-auto mb-2 text-blue-600" size={32} />
              <p className="font-semibold">Exchange API</p>
              <p className="text-xs text-gray-600">Real-time OHLCV</p>
            </div>
          </div>
          <div className="text-2xl text-gray-400">→</div>
          <div className="flex-1">
            <div className="bg-green-100 rounded-lg p-4 mb-2">
              <Activity className="mx-auto mb-2 text-green-600" size={32} />
              <p className="font-semibold">ML Models</p>
              <p className="text-xs text-gray-600">GARCH + LSTM</p>
            </div>
          </div>
          <div className="text-2xl text-gray-400">→</div>
          <div className="flex-1">
            <div className="bg-purple-100 rounded-lg p-4 mb-2">
              <Cloud className="mx-auto mb-2 text-purple-600" size={32} />
              <p className="font-semibold">Prediction API</p>
              <p className="text-xs text-gray-600">24h Volatility</p>
            </div>
          </div>
          <div className="text-2xl text-gray-400">→</div>
          <div className="flex-1">
            <div className="bg-orange-100 rounded-lg p-4 mb-2">
              <Bell className="mx-auto mb-2 text-orange-600" size={32} />
              <p className="font-semibold">Alerts & Dashboard</p>
              <p className="text-xs text-gray-600">User Notifications</p>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-6 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-xl shadow-lg p-6 text-white">
        <h3 className="text-xl font-bold mb-3">🎯 Project Timeline</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="font-semibold mb-1">Week 1: Data Pipeline</p>
            <p className="text-indigo-100">API setup, feature engineering, storage</p>
          </div>
          <div>
            <p className="font-semibold mb-1">Week 2-3: ML Development</p>
            <p className="text-indigo-100">Train GARCH & LSTM, compare models</p>
          </div>
          <div>
            <p className="font-semibold mb-1">Week 4: MLOps Infrastructure</p>
            <p className="text-indigo-100">Docker, CI/CD, deployment</p>
          </div>
          <div>
            <p className="font-semibold mb-1">Week 5: Monitoring & Polish</p>
            <p className="text-indigo-100">Dashboard, alerts, documentation</p>
          </div>
        </div>
      </div>

      <div className="mt-4 text-center text-sm text-gray-500">
        Click on any component to see detailed information
      </div>
    </div>
  );
};

export default ArchitectureDiagram;