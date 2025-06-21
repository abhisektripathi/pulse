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
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
} from '@mui/material';
import {
  TrendingUp as TrendingIcon,
  Warning as WarningIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';

interface PredictionData {
  entity_id: string;
  prediction_type: string;
  confidence: number;
  predicted_value: number;
  timestamp: string;
  model_info?: {
    model_name: string;
    version: string;
    accuracy: number;
  };
}

const Predictions = () => {
  const [predictions, setPredictions] = useState<PredictionData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [entityId, setEntityId] = useState('payment-gateway');
  const [predictionType, setPredictionType] = useState('failure_probability');
  const [timeHorizon, setTimeHorizon] = useState(24);

  const fetchPredictions = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          entity_id: entityId,
          prediction_type: predictionType,
          time_horizon: timeHorizon
        }),
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Prediction data received:', data); // Debug log
        setPredictions([data]); // Convert single prediction to array for consistency
      } else {
        setError('Failed to fetch predictions');
      }
    } catch (err) {
      setError('Failed to connect to backend');
      console.error('Error fetching predictions:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPredictions();
  }, [entityId, predictionType, timeHorizon]);

  const getPredictionColor = (value: number, type: string) => {
    if (type === 'failure_probability') {
      if (value >= 0.7) return 'error';
      if (value >= 0.4) return 'warning';
      return 'success';
    }
    return 'primary';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  const formatPredictionValue = (value: number, type: string) => {
    if (value === undefined || value === null || isNaN(value)) {
      return 'N/A';
    }
    if (type === 'failure_probability') {
      return `${(value * 100).toFixed(1)}%`;
    }
    return value.toFixed(2);
  };

  const getPredictionIcon = (value: number, type: string) => {
    if (value === undefined || value === null || isNaN(value)) {
      return <TrendingIcon color="primary" />;
    }
    if (type === 'failure_probability') {
      if (value >= 0.7) return <ErrorIcon color="error" />;
      if (value >= 0.4) return <WarningIcon color="warning" />;
      return <CheckIcon color="success" />;
    }
    return <TrendingIcon color="primary" />;
  };

  const safeToFixed = (value: number | undefined, decimals: number = 1): string => {
    if (value === undefined || value === null || isNaN(value)) {
      return 'N/A';
    }
    return value.toFixed(decimals);
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        ML Predictions
      </Typography>

      {/* Prediction Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Prediction Parameters
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                label="Entity ID"
                value={entityId}
                onChange={(e) => setEntityId(e.target.value)}
                variant="outlined"
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Prediction Type</InputLabel>
                <Select
                  value={predictionType}
                  label="Prediction Type"
                  onChange={(e) => setPredictionType(e.target.value)}
                >
                  <MenuItem value="failure_probability">Failure Probability</MenuItem>
                  <MenuItem value="performance_forecast">Performance Forecast</MenuItem>
                  <MenuItem value="anomaly_detection">Anomaly Detection</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                label="Time Horizon (hours)"
                type="number"
                value={timeHorizon}
                onChange={(e) => setTimeHorizon(Number(e.target.value))}
                variant="outlined"
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <Button
                variant="contained"
                onClick={fetchPredictions}
                disabled={loading}
                fullWidth
                sx={{ height: 56 }}
              >
                {loading ? <CircularProgress size={24} /> : 'Refresh Predictions'}
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {loading ? (
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
          <CircularProgress />
        </Box>
      ) : predictions.length > 0 ? (
        <Grid container spacing={3}>
          {predictions.map((prediction, index) => (
            <React.Fragment key={index}>
              {/* Main Prediction Card */}
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      {getPredictionIcon(prediction.predicted_value, prediction.prediction_type)}
                      <Typography variant="h6" ml={1}>
                        {prediction.prediction_type.replace('_', ' ').toUpperCase()}
                      </Typography>
                    </Box>
                    
                    <Typography variant="h2" 
                      color={getPredictionColor(prediction.predicted_value, prediction.prediction_type) as any}
                      gutterBottom
                    >
                      {formatPredictionValue(prediction.predicted_value, prediction.prediction_type)}
                    </Typography>
                    
                    <Box display="flex" gap={1} mb={2}>
                      <Chip 
                        label={`Confidence: ${safeToFixed(prediction.confidence * 100)}%`}
                        color={getConfidenceColor(prediction.confidence) as any}
                        size="small"
                      />
                      <Chip 
                        label={`Entity: ${prediction.entity_id}`}
                        variant="outlined"
                        size="small"
                      />
                    </Box>
                    
                    <Typography variant="body2" color="text.secondary">
                      Predicted at: {prediction.timestamp ? new Date(prediction.timestamp).toLocaleString() : 'Unknown'}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              {/* Model Information */}
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Model Information
                    </Typography>
                    
                    {prediction.model_info ? (
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">
                            Model Name
                          </Typography>
                          <Typography variant="body1">
                            {prediction.model_info.model_name || 'Unknown'}
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">
                            Version
                          </Typography>
                          <Typography variant="body1">
                            {prediction.model_info.version || 'Unknown'}
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">
                            Accuracy
                          </Typography>
                          <Typography variant="body1">
                            {safeToFixed(prediction.model_info.accuracy * 100)}%
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">
                            Prediction Type
                          </Typography>
                          <Typography variant="body1">
                            {prediction.prediction_type}
                          </Typography>
                        </Grid>
                      </Grid>
                    ) : (
                      <Typography color="text.secondary">
                        Model information not available
                      </Typography>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* Prediction Details */}
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Prediction Analysis
                    </Typography>
                    
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={4}>
                        <Box textAlign="center">
                          <Typography variant="h4" color="primary">
                            {formatPredictionValue(prediction.predicted_value, prediction.prediction_type)}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Predicted Value
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Box textAlign="center">
                          <Typography variant="h4" color="secondary">
                            {safeToFixed(prediction.confidence * 100)}%
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Model Confidence
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Box textAlign="center">
                          <Typography variant="h4" color="info.main">
                            {timeHorizon}h
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Time Horizon
                          </Typography>
                        </Box>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </React.Fragment>
          ))}
        </Grid>
      ) : (
        <Alert severity="info">
          No predictions available. Please check the prediction parameters and try again.
        </Alert>
      )}
    </Box>
  );
};

export default Predictions; 