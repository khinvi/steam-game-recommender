import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  CircularProgress,
  Alert,
  AppBar,
  Toolbar,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Paper,
  Avatar,
  ListItemAvatar,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  Gamepad as GamepadIcon,
  Dashboard as DashboardIcon,
  Recommend as RecommendIcon,
  BarChart as StatsIcon,
  Info as AboutIcon,
  Menu as MenuIcon,
  Close as CloseIcon,
  Star as StarIcon,
  TrendingUp as TrendingIcon,
  Person as PersonIcon,
  Speed as SpeedIcon,
  DataUsage as DataIcon,
  ExpandMore as ExpandMoreIcon,
  Psychology as PsychologyIcon,
  Group as GroupIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';
import axios from 'axios';

interface Game {
  game_id: string;
  game_name: string;
  genres: string[];
  tags: string[];
  price: number;
  score?: number;
  score_breakdown?: any;
}

interface RecommendationResponse {
  user_id: string;
  recommendations: Game[];
  model_used: string;
  total_recommendations: number;
}

interface SampleUser {
  id: string;
  reviews: number;
  description: string;
  genre: string;
}

interface ModelInfo {
  models: string[];
  performance: Record<string, string>;
  dataset_stats: {
    total_users: number;
    total_games: number;
    total_interactions: number;
    data_source: string;
  };
}

const API_BASE = 'http://localhost:8000';

export default function SteamGameRecommender() {
  const [activePage, setActivePage] = useState('dashboard');
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [games, setGames] = useState<Game[]>([]);
  const [recommendations, setRecommendations] = useState<Game[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  // Form state
  const [selectedUser, setSelectedUser] = useState<string>('');
  const [modelType, setModelType] = useState('svd');
  const [numRecs, setNumRecs] = useState(10);
  const [includePlayed, setIncludePlayed] = useState('false');
  
  // New state for sample users and model info
  const [sampleUsers, setSampleUsers] = useState<SampleUser[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [showUserSelector, setShowUserSelector] = useState(false);

  useEffect(() => {
    loadGames();
    loadSampleUsers();
    loadModelInfo();
  }, []);

  const loadGames = async () => {
    try {
      const response = await axios.get<Game[]>(`${API_BASE}/games`);
      setGames(response.data);
    } catch (err) {
      console.error('Error loading games:', err);
    }
  };

  const loadSampleUsers = async () => {
    try {
      const response = await axios.get<SampleUser[]>(`${API_BASE}/recommendations/sample-users`);
      setSampleUsers(response.data);
      if (response.data.length > 0) {
        setSelectedUser(response.data[0].id);
      }
    } catch (err) {
      console.error('Error loading sample users:', err);
      // Create fallback sample users
      setSampleUsers([
        { id: "user_12345", reviews: 45, description: "RPG enthusiast", genre: "RPG" },
        { id: "user_67890", reviews: 120, description: "Indie game lover", genre: "Indie" },
        { id: "user_24680", reviews: 78, description: "Action gamer", genre: "Action" },
        { id: "user_13579", reviews: 95, description: "Strategy master", genre: "Strategy" },
        { id: "user_97531", reviews: 62, description: "Adventure seeker", genre: "Adventure" }
      ]);
      setSelectedUser("user_12345");
    }
  };

  const loadModelInfo = async () => {
    try {
      const response = await axios.get<ModelInfo>(`${API_BASE}/recommendations/models`);
      setModelInfo(response.data);
    } catch (err) {
      console.error('Error loading model info:', err);
    }
  };

  const getRecommendations = async () => {
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      const response = await axios.post<RecommendationResponse>(`${API_BASE}/recommendations/generate`, {
        user_id: selectedUser,
        n_recommendations: numRecs,
        model_type: modelType,
        include_played: includePlayed === 'true'
      });
      
      setRecommendations(response.data.recommendations);
      setSuccess(`Generated ${response.data.total_recommendations} recommendations using ${response.data.model_used} model`);
      setActivePage('recommendations');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to get recommendations');
    } finally {
      setLoading(false);
    }
  };

  const renderDashboard = () => (
    <Box>
      <Typography variant="h4" gutterBottom>
        🎮 Steam Game Recommender
      </Typography>
      
      <Typography variant="body1" color="text.secondary" paragraph>
        Get personalized game recommendations using advanced AI models trained on the Stanford SNAP Steam dataset.
      </Typography>

      {/* Dataset Statistics */}
      {modelInfo && (
        <Card sx={{ mb: 3, bgcolor: 'primary.50' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              📊 Dataset Statistics
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6} md={3}>
                <Typography variant="h4" color="primary">
                  {modelInfo.dataset_stats.total_users.toLocaleString()}
                </Typography>
                <Typography variant="body2">Active Users</Typography>
              </Grid>
              <Grid item xs={6} md={3}>
                <Typography variant="h4" color="primary">
                  {modelInfo.dataset_stats.total_games.toLocaleString()}
                </Typography>
                <Typography variant="body2">Games Available</Typography>
              </Grid>
              <Grid item xs={6} md={3}>
                <Typography variant="h4" color="primary">
                  {modelInfo.dataset_stats.total_interactions.toLocaleString()}
                </Typography>
                <Typography variant="body2">Reviews & Ratings</Typography>
              </Grid>
              <Grid item xs={6} md={3}>
                <Typography variant="h4" color="primary">
                  {modelInfo.models.length}
                </Typography>
                <Typography variant="body2">AI Models</Typography>
              </Grid>
            </Grid>
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              Data Source: {modelInfo.dataset_stats.data_source}
            </Typography>
          </CardContent>
        </Card>
      )}

      {/* Model Performance */}
      {modelInfo && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              🚀 Model Performance
            </Typography>
            <Grid container spacing={2}>
              {Object.entries(modelInfo.performance).map(([model, description]) => (
                <Grid item xs={12} md={6} key={model}>
                  <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                    <Typography variant="subtitle1" fontWeight="bold" color="primary">
                      {model.toUpperCase()}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {description}
                    </Typography>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Quick Start */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🚀 Quick Start
          </Typography>
          <Typography variant="body2" paragraph>
            Choose a sample user and get instant recommendations using our best-performing SVD model.
          </Typography>
          
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={4}>
              <FormControl fullWidth>
                <InputLabel>Select User</InputLabel>
                <Select
                  value={selectedUser}
                  onChange={(e) => setSelectedUser(e.target.value)}
                  label="Select User"
                >
                  {sampleUsers.map((user) => (
                    <MenuItem key={user.id} value={user.id}>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Avatar sx={{ width: 32, height: 32, mr: 1, bgcolor: 'primary.main' }}>
                          {user.genre.charAt(0)}
                        </Avatar>
                        <Box>
                          <Typography variant="body2" fontWeight="bold">
                            {user.description}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {user.reviews} reviews
                          </Typography>
                        </Box>
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <FormControl fullWidth>
                <InputLabel>Model</InputLabel>
                <Select
                  value={modelType}
                  onChange={(e) => setModelType(e.target.value)}
                  label="Model"
                >
                  <MenuItem value="svd">SVD (Best Accuracy - 26% Precision)</MenuItem>
                  <MenuItem value="item_based">Item-Based CF (Fast & Simple)</MenuItem>
                  <MenuItem value="popularity">Trending Games (Baseline)</MenuItem>
                  <MenuItem value="hybrid">Hybrid (Balanced)</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Button
                variant="contained"
                size="large"
                onClick={getRecommendations}
                disabled={loading || !selectedUser}
                startIcon={<RecommendIcon />}
                fullWidth
              >
                {loading ? <CircularProgress size={20} /> : 'Get Recommendations'}
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );

  const renderRecommendations = () => (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          🎯 Your Recommendations
        </Typography>
        <Button
          variant="outlined"
          onClick={() => setActivePage('dashboard')}
          startIcon={<DashboardIcon />}
        >
          Back to Dashboard
        </Button>
      </Box>

      {success && (
        <Alert severity="success" sx={{ mb: 3 }}>
          {success}
        </Alert>
      )}

      {recommendations.length > 0 ? (
        <Grid container spacing={3}>
          {recommendations.map((game, index) => (
            <Grid item xs={12} md={6} lg={4} key={game.game_id}>
              <Card sx={{ height: '100%', position: 'relative' }}>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                    <Typography variant="h6" component="div" sx={{ flex: 1 }}>
                      {game.game_name}
                    </Typography>
                    {game.score && (
                      <Chip
                        label={`${(game.score * 100).toFixed(0)}%`}
                        color="primary"
                        size="small"
                        icon={<StarIcon />}
                      />
                    )}
                  </Box>
                  
                  <Typography variant="body2" color="text.secondary" paragraph>
                    {game.genres?.join(', ') || 'No genre specified'}
                  </Typography>
                  
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="h6" color="primary">
                      ${game.price?.toFixed(2) || 'Free'}
                    </Typography>
                    <Chip
                      label={`#${index + 1}`}
                      color="secondary"
                      size="small"
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      ) : (
        <Card>
          <CardContent sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="h6" color="text.secondary">
              No recommendations yet
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Click "Get Recommendations" to start
            </Typography>
          </CardContent>
        </Card>
      )}
    </Box>
  );

  const renderAdvancedSettings = () => (
    <Box>
      <Typography variant="h4" gutterBottom>
        ⚙️ Advanced Settings
      </Typography>
      
      <Card>
        <CardContent>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Number of Recommendations</InputLabel>
                <Select
                  value={numRecs}
                  onChange={(e) => setNumRecs(e.target.value as number)}
                  label="Number of Recommendations"
                >
                  <MenuItem value={5}>5 (High Quality)</MenuItem>
                  <MenuItem value={10}>10 (Balanced)</MenuItem>
                  <MenuItem value={15}>15 (More Options)</MenuItem>
                  <MenuItem value={20}>20 (Maximum)</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Include Played Games</InputLabel>
                <Select
                  value={includePlayed}
                  onChange={(e) => setIncludePlayed(e.target.value)}
                  label="Include Played Games"
                >
                  <MenuItem value="false">No (New Games Only)</MenuItem>
                  <MenuItem value="true">Yes (All Games)</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
          
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Model Selection Guide
            </Typography>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">SVD (Matrix Factorization)</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography variant="body2">
                  <strong>Best for:</strong> Overall accuracy and personalized recommendations<br/>
                  <strong>Performance:</strong> 26% precision - our top performer<br/>
                  <strong>Use when:</strong> You want the most accurate recommendations
                </Typography>
              </AccordionDetails>
            </Accordion>
            
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">Item-Based Collaborative Filtering</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography variant="body2">
                  <strong>Best for:</strong> Fast recommendations and finding similar games<br/>
                  <strong>Performance:</strong> 8-12% precision<br/>
                  <strong>Use when:</strong> You want quick results or are exploring similar games
                </Typography>
              </AccordionDetails>
            </Accordion>
            
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">Popularity-Based</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography variant="body2">
                  <strong>Best for:</strong> New users and trending games<br/>
                  <strong>Performance:</strong> 5-8% precision<br/>
                  <strong>Use when:</strong> You're new or want to see what's popular
                </Typography>
              </AccordionDetails>
            </Accordion>
            
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">Hybrid</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography variant="body2">
                  <strong>Best for:</strong> Balanced approach combining multiple models<br/>
                  <strong>Performance:</strong> 15-20% precision<br/>
                  <strong>Use when:</strong> You want a mix of accuracy and diversity
                </Typography>
              </AccordionDetails>
            </Accordion>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );

  const renderContent = () => {
    switch (activePage) {
      case 'dashboard':
        return renderDashboard();
      case 'recommendations':
        return renderRecommendations();
      case 'advanced':
        return renderAdvancedSettings();
      default:
        return renderDashboard();
    }
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <IconButton
            edge="start"
            color="inherit"
            onClick={() => setDrawerOpen(true)}
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          <GamepadIcon sx={{ mr: 1 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Steam Game Recommender
          </Typography>
        </Toolbar>
      </AppBar>

      <Drawer
        anchor="left"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      >
        <Box sx={{ width: 250 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', p: 2 }}>
            <GamepadIcon sx={{ mr: 1 }} />
            <Typography variant="h6">Menu</Typography>
            <IconButton
              onClick={() => setDrawerOpen(false)}
              sx={{ ml: 'auto' }}
            >
              <CloseIcon />
            </IconButton>
          </Box>
          <Divider />
          <List>
            <ListItem button onClick={() => { setActivePage('dashboard'); setDrawerOpen(false); }}>
              <ListItemIcon><DashboardIcon /></ListItemIcon>
              <ListItemText primary="Dashboard" />
            </ListItem>
            <ListItem button onClick={() => { setActivePage('recommendations'); setDrawerOpen(false); }}>
              <ListItemIcon><RecommendIcon /></ListItemIcon>
              <ListItemText primary="Recommendations" />
            </ListItem>
            <ListItem button onClick={() => { setActivePage('advanced'); setDrawerOpen(false); }}>
              <ListItemIcon><SettingsIcon /></ListItemIcon>
              <ListItemText primary="Advanced Settings" />
            </ListItem>
          </List>
        </Box>
      </Drawer>

      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}
        
        {renderContent()}
      </Container>
    </Box>
  );
} 