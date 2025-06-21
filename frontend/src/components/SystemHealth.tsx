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
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import {
  HealthAndSafety as HealthIcon,
  Warning as WarningIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
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
  alerts?: Array<{
    severity: string;
    message: string;
    timestamp: string;
  }>;
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
    alerts: [], // Backend doesn't provide alerts in current response
  };
}

const SystemHealth = () => {
  const [healthData, setHealthData] = useState<HealthData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHealthData = async () => {
      try {
        setLoading(true);
        const response = await fetch('http://localhost:8000/health/payment-gateway');
        if (response.ok) {
          const data = await response.json();
          console.log('Raw health data received:', data); // Debug log
          const mappedData = mapApiToHealthData(data);
          console.log('Mapped health data:', mappedData); // Debug log
          setHealthData(mappedData);
        } else {
          setError('Failed to fetch health data');
        }
      } catch (err) {
        setError('Failed to connect to backend');
        console.error('Error fetching health data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchHealthData();
    const interval = setInterval(fetchHealthData, 30000); // Refresh every 30 seconds

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

  const getSeverityColor = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'critical':
        return 'error';
      case 'warning':
        return 'warning';
      case 'info':
        return 'info';
      default:
        return 'default';
    }
  };

  const getMetricColor = (value: number, threshold: number) => {
    if (!value || isNaN(value)) return 'default';
    if (value >= threshold) return 'error';
    if (value >= threshold * 0.8) return 'warning';
    return 'success';
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

  if (!healthData) {
    return (
      <Alert severity="info">
        No health data available. Please check if the backend server is running.
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        System Health Monitoring
      </Typography>

      <Grid container spacing={3}>
        {/* Overall Health Score */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <HealthIcon color="primary" />
                <Typography variant="h6" ml={1}>
                  Overall Health Score
                </Typography>
              </Box>
              <Typography variant="h2" color="primary" gutterBottom>
                {safeToFixed(healthData.health_score)}%
              </Typography>
              <Chip 
                label={healthData.status || 'Unknown'} 
                color={getStatusColor(healthData.status) as any}
                sx={{ mb: 1 }}
              />
              <Typography variant="body2" color="text.secondary">
                Last updated: {healthData.last_updated ? new Date(healthData.last_updated).toLocaleString() : 'Unknown'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* CPU Usage */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                CPU Usage
              </Typography>
              <Box display="flex" alignItems="center" mb={1}>
                <Typography variant="h4" color={getMetricColor(healthData.metrics?.cpu_usage, 80) as any}>
                  {safeToFixed(healthData.metrics?.cpu_usage)}%
                </Typography>
              </Box>
              <LinearProgress 
                variant="determinate" 
                value={healthData.metrics?.cpu_usage || 0} 
                color={getMetricColor(healthData.metrics?.cpu_usage, 80) as any}
                sx={{ height: 8, borderRadius: 4 }}
              />
              <Typography variant="body2" color="text.secondary" mt={1}>
                Threshold: 80%
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Memory Usage */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Memory Usage
              </Typography>
              <Box display="flex" alignItems="center" mb={1}>
                <Typography variant="h4" color={getMetricColor(healthData.metrics?.memory_usage, 85) as any}>
                  {safeToFixed(healthData.metrics?.memory_usage)}%
                </Typography>
              </Box>
              <LinearProgress 
                variant="determinate" 
                value={healthData.metrics?.memory_usage || 0} 
                color={getMetricColor(healthData.metrics?.memory_usage, 85) as any}
                sx={{ height: 8, borderRadius: 4 }}
              />
              <Typography variant="body2" color="text.secondary" mt={1}>
                Threshold: 85%
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Detailed Metrics Table */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Detailed Metrics
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Metric</TableCell>
                      <TableCell>Current Value</TableCell>
                      <TableCell>Threshold</TableCell>
                      <TableCell>Status</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell>CPU Usage</TableCell>
                      <TableCell>{safeToFixed(healthData.metrics?.cpu_usage)}%</TableCell>
                      <TableCell>80%</TableCell>
                      <TableCell>
                        <Chip 
                          label={
                            !healthData.metrics?.cpu_usage ? 'Unknown' :
                            healthData.metrics.cpu_usage >= 80 ? 'Critical' : 
                            healthData.metrics.cpu_usage >= 64 ? 'Warning' : 'Normal'
                          }
                          color={getMetricColor(healthData.metrics?.cpu_usage, 80) as any}
                          size="small"
                        />
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Memory Usage</TableCell>
                      <TableCell>{safeToFixed(healthData.metrics?.memory_usage)}%</TableCell>
                      <TableCell>85%</TableCell>
                      <TableCell>
                        <Chip 
                          label={
                            !healthData.metrics?.memory_usage ? 'Unknown' :
                            healthData.metrics.memory_usage >= 85 ? 'Critical' : 
                            healthData.metrics.memory_usage >= 68 ? 'Warning' : 'Normal'
                          }
                          color={getMetricColor(healthData.metrics?.memory_usage, 85) as any}
                          size="small"
                        />
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Response Time</TableCell>
                      <TableCell>{safeToFixed(healthData.metrics?.response_time, 2)}ms</TableCell>
                      <TableCell>500ms</TableCell>
                      <TableCell>
                        <Chip 
                          label={
                            !healthData.metrics?.response_time ? 'Unknown' :
                            healthData.metrics.response_time >= 500 ? 'Critical' : 
                            healthData.metrics.response_time >= 400 ? 'Warning' : 'Normal'
                          }
                          color={getMetricColor(healthData.metrics?.response_time, 500) as any}
                          size="small"
                        />
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Error Rate</TableCell>
                      <TableCell>{safeToFixed(healthData.metrics?.error_rate, 3)}%</TableCell>
                      <TableCell>1%</TableCell>
                      <TableCell>
                        <Chip 
                          label={
                            !healthData.metrics?.error_rate ? 'Unknown' :
                            healthData.metrics.error_rate >= 1 ? 'Critical' : 
                            healthData.metrics.error_rate >= 0.8 ? 'Warning' : 'Normal'
                          }
                          color={getMetricColor(healthData.metrics?.error_rate, 1) as any}
                          size="small"
                        />
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Alerts */}
        {healthData.alerts && healthData.alerts.length > 0 && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Active Alerts
                </Typography>
                <Box>
                  {healthData.alerts.map((alert, index) => (
                    <Alert 
                      key={index}
                      severity={getSeverityColor(alert.severity) as any}
                      sx={{ mb: 1 }}
                    >
                      <Typography variant="body2">
                        {alert.message}
                      </Typography>
                      <Typography variant="caption" display="block">
                        {alert.timestamp ? new Date(alert.timestamp).toLocaleString() : 'Unknown time'}
                      </Typography>
                    </Alert>
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default SystemHealth; 