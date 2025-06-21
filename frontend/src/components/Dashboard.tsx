import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
  Chip,
} from '@mui/material';
import {
  HealthAndSafety as HealthIcon,
  TrendingUp as TrendingIcon,
  Warning as WarningIcon,
  CheckCircle as CheckIcon,
} from '@mui/icons-material';

interface HealthData {
  entity_id: string;
  health_score: number;
  status: string;
  last_updated: string;
  metrics: {
    cpu_usage: number;
    memory_usage: number;
    response_time: number;
    error_rate: number;
  };
}

interface PredictionData {
  entity_id: string;
  prediction_type: string;
  confidence: number;
  predicted_value: number;
  timestamp: string;
}

// Mapping function to convert backend API response to frontend format
function mapApiToHealthData(apiResponse: any): HealthData {
  const resourceUtilization = apiResponse.factors?.find((f: any) => f.name === 'resource_utilization')?.value ?? 0;
  const performance = apiResponse.factors?.find((f: any) => f.name === 'performance')?.value ?? 0;
  const errorRate = apiResponse.factors?.find((f: any) => f.name === 'error_rate')?.value ?? 0;
  
  return {
    entity_id: apiResponse.entity_id || 'unknown',
    health_score: apiResponse.score || 0,
    status: apiResponse.category || 'unknown',
    last_updated: apiResponse.last_updated || apiResponse.timestamp || new Date().toISOString(),
    metrics: {
      cpu_usage: resourceUtilization,
      memory_usage: resourceUtilization,
      response_time: performance,
      error_rate: errorRate,
    },
  };
}

const Dashboard = () => {
  const [healthData, setHealthData] = useState<HealthData | null>(null);
  const [predictionData, setPredictionData] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Fetch health data
        const healthResponse = await fetch('http://localhost:8000/health/payment-gateway');
        if (healthResponse.ok) {
          const health = await healthResponse.json();
          console.log('Raw health data received:', health); // Debug log
          const mappedHealth = mapApiToHealthData(health);
          console.log('Mapped health data:', mappedHealth); // Debug log
          setHealthData(mappedHealth);
        }

        // Fetch prediction data
        const predictionResponse = await fetch('http://localhost:8000/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            entity_id: 'payment-gateway',
            prediction_type: 'failure_probability',
            time_horizon: 24
          }),
        });
        
        if (predictionResponse.ok) {
          const prediction = await predictionResponse.json();
          console.log('Prediction data received:', prediction); // Debug log
          setPredictionData(prediction);
        }
      } catch (err) {
        setError('Failed to fetch data from backend');
        console.error('Error fetching data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'healthy':
        return 'success';
      case 'degraded':
      case 'warning':
        return 'warning';
      case 'critical':
        return 'error';
      default:
        return 'default';
    }
  };

  const getHealthIcon = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'healthy':
        return <CheckIcon color="success" />;
      case 'degraded':
      case 'warning':
        return <WarningIcon color="warning" />;
      case 'critical':
        return <WarningIcon color="error" />;
      default:
        return <HealthIcon />;
    }
  };

  const safeToFixed = (value: number | undefined, decimals: number = 1): string => {
    if (value === undefined || value === null || isNaN(value)) {
      return 'N/A';
    }
    return value.toFixed(decimals);
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        System Health Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        {/* Health Score Card */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                {getHealthIcon(healthData?.status || 'unknown')}
                <Typography variant="h6" ml={1}>
                  System Health Score
                </Typography>
              </Box>
              
              {healthData ? (
                <>
                  <Typography variant="h3" color="primary" gutterBottom>
                    {safeToFixed(healthData.health_score)}%
                  </Typography>
                  <Chip 
                    label={healthData.status} 
                    color={getStatusColor(healthData.status) as any}
                    sx={{ mb: 2 }}
                  />
                  <Typography variant="body2" color="text.secondary">
                    Last updated: {healthData.last_updated ? new Date(healthData.last_updated).toLocaleString() : 'Unknown'}
                  </Typography>
                </>
              ) : (
                <Typography color="text.secondary">No health data available</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Prediction Card */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <TrendingIcon color="primary" />
                <Typography variant="h6" ml={1}>
                  Failure Prediction
                </Typography>
              </Box>
              
              {predictionData ? (
                <>
                  <Typography variant="h3" color="secondary" gutterBottom>
                    {(predictionData.predicted_value * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Confidence: {(predictionData.confidence * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Predicted: {predictionData.prediction_type}
                  </Typography>
                </>
              ) : (
                <Typography color="text.secondary">No prediction data available</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Metrics Grid */}
        {healthData && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  System Metrics
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6} md={3}>
                    <Typography variant="body2" color="text.secondary">CPU Usage</Typography>
                    <Typography variant="h6">{safeToFixed(healthData.metrics.cpu_usage)}%</Typography>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Typography variant="body2" color="text.secondary">Memory Usage</Typography>
                    <Typography variant="h6">{safeToFixed(healthData.metrics.memory_usage)}%</Typography>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Typography variant="body2" color="text.secondary">Response Time</Typography>
                    <Typography variant="h6">{safeToFixed(healthData.metrics.response_time, 2)}ms</Typography>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Typography variant="body2" color="text.secondary">Error Rate</Typography>
                    <Typography variant="h6">{safeToFixed(healthData.metrics.error_rate, 3)}%</Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default Dashboard; 