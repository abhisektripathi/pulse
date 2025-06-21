import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Typography,
  Box,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  HealthAndSafety as HealthIcon,
  TrendingUp as PredictionsIcon,
  Chat as ChatIcon,
} from '@mui/icons-material';

const drawerWidth = 240;

const Navigation = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
    { text: 'System Health', icon: <HealthIcon />, path: '/health' },
    { text: 'Predictions', icon: <PredictionsIcon />, path: '/predictions' },
    { text: 'Conversation', icon: <ChatIcon />, path: '/conversation' },
  ];

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
          backgroundColor: '#1a1a1a',
          borderRight: '1px solid #333',
        },
      }}
    >
      <Box sx={{ p: 2, borderBottom: '1px solid #333' }}>
        <Typography variant="h6" color="primary" sx={{ fontWeight: 'bold' }}>
          Predictive Health
        </Typography>
        <Typography variant="caption" color="text.secondary">
          System Monitoring
        </Typography>
      </Box>
      <List sx={{ pt: 1 }}>
        {menuItems.map((item) => (
          <ListItem
            button
            key={item.text}
            onClick={() => navigate(item.path)}
            selected={location.pathname === item.path}
            sx={{
              '&.Mui-selected': {
                backgroundColor: 'rgba(33, 150, 243, 0.1)',
                borderRight: '3px solid #2196f3',
              },
              '&:hover': {
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
              },
            }}
          >
            <ListItemIcon sx={{ color: 'inherit' }}>
              {item.icon}
            </ListItemIcon>
            <ListItemText primary={item.text} />
          </ListItem>
        ))}
      </List>
    </Drawer>
  );
};

export default Navigation; 