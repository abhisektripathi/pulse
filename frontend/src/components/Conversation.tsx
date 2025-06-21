import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  CircularProgress,
  Alert,
  Paper,
  Avatar,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Divider,
  Chip,
  Grid,
} from '@mui/material';
import {
  Send as SendIcon,
  SmartToy as BotIcon,
  Person as UserIcon,
} from '@mui/icons-material';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  confidence?: number;
}

const Conversation = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hello! I\'m your AI assistant for the Predictive System Health Platform. I can help you with system health queries, predictions, and troubleshooting. What would you like to know?',
      sender: 'bot',
      timestamp: new Date(),
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputText,
          context: {
            user_id: 'demo_user',
            session_id: 'demo_session',
            timestamp: new Date().toISOString(),
          }
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const botMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: data.response,
          sender: 'bot',
          timestamp: new Date(),
          confidence: data.confidence,
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        throw new Error('Failed to get response from AI');
      }
    } catch (err) {
      setError('Failed to connect to AI service. Please try again.');
      console.error('Error sending message:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  const getSuggestedQuestions = () => [
    'What is the current health status of the payment gateway?',
    'Show me the latest predictions for system failures',
    'What are the main performance metrics?',
    'Are there any active alerts or issues?',
    'How can I improve system reliability?',
  ];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        AI Assistant
      </Typography>

      <Grid container spacing={3}>
        {/* Chat Interface */}
        <Grid item xs={12} md={8}>
          <Card sx={{ height: '70vh', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column', p: 0 }}>
              {/* Messages Area */}
              <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
                <List>
                  {messages.map((message) => (
                    <React.Fragment key={message.id}>
                      <ListItem
                        sx={{
                          flexDirection: 'column',
                          alignItems: message.sender === 'user' ? 'flex-end' : 'flex-start',
                        }}
                      >
                        <Box
                          sx={{
                            display: 'flex',
                            alignItems: 'flex-start',
                            maxWidth: '70%',
                            flexDirection: message.sender === 'user' ? 'row-reverse' : 'row',
                          }}
                        >
                          <ListItemAvatar sx={{ minWidth: 40 }}>
                            <Avatar sx={{ bgcolor: message.sender === 'user' ? 'primary.main' : 'secondary.main' }}>
                              {message.sender === 'user' ? <UserIcon /> : <BotIcon />}
                            </Avatar>
                          </ListItemAvatar>
                          <Paper
                            sx={{
                              p: 2,
                              bgcolor: message.sender === 'user' ? 'primary.main' : 'grey.100',
                              color: message.sender === 'user' ? 'white' : 'text.primary',
                              borderRadius: 2,
                            }}
                          >
                            <ListItemText
                              primary={message.text}
                              secondary={
                                <Box sx={{ mt: 1 }}>
                                  <Typography variant="caption" color="text.secondary">
                                    {message.timestamp.toLocaleTimeString()}
                                  </Typography>
                                  {message.confidence && (
                                    <Chip
                                      label={`Confidence: ${(message.confidence * 100).toFixed(1)}%`}
                                      size="small"
                                      sx={{ ml: 1 }}
                                    />
                                  )}
                                </Box>
                              }
                            />
                          </Paper>
                        </Box>
                      </ListItem>
                      <Divider />
                    </React.Fragment>
                  ))}
                  {loading && (
                    <ListItem>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <CircularProgress size={20} />
                        <Typography variant="body2" color="text.secondary">
                          AI is thinking...
                        </Typography>
                      </Box>
                    </ListItem>
                  )}
                  <div ref={messagesEndRef} />
                </List>
              </Box>

              {/* Input Area */}
              <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
                {error && (
                  <Alert severity="error" sx={{ mb: 2 }}>
                    {error}
                  </Alert>
                )}
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <TextField
                    fullWidth
                    multiline
                    maxRows={3}
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask me about system health, predictions, or troubleshooting..."
                    variant="outlined"
                    disabled={loading}
                  />
                  <Button
                    variant="contained"
                    onClick={sendMessage}
                    disabled={loading || !inputText.trim()}
                    sx={{ minWidth: 56 }}
                  >
                    <SendIcon />
                  </Button>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Suggested Questions */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Suggested Questions
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                {getSuggestedQuestions().map((question, index) => (
                  <Button
                    key={index}
                    variant="outlined"
                    size="small"
                    onClick={() => setInputText(question)}
                    sx={{ justifyContent: 'flex-start', textAlign: 'left' }}
                  >
                    {question}
                  </Button>
                ))}
              </Box>
            </CardContent>
          </Card>

          {/* AI Capabilities */}
          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                AI Capabilities
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Chip label="System Health Analysis" size="small" />
                <Chip label="Failure Predictions" size="small" />
                <Chip label="Performance Monitoring" size="small" />
                <Chip label="Troubleshooting Guidance" size="small" />
                <Chip label="Alert Management" size="small" />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Conversation; 