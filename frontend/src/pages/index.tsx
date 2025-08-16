import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Button,
  TextField,
  Chip,
  Rating,
  Alert,
  CircularProgress,
  AppBar,
  Toolbar,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider
} from '@mui/material';
import {
  Search as SearchIcon,
  Gamepad as GamepadIcon,
  Star as StarIcon,
  Person as PersonIcon,
  Menu as MenuIcon,
  Home as HomeIcon,
  TrendingUp as TrendingIcon,
  Favorite as FavoriteIcon
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

const API_BASE = 'http://localhost:8000';

export default function HomePage() {
  const [games, setGames] = useState<Game[]>([]);
  const [recommendations, setRecommendations] = useState<Game[]>([]);
  const [popularGames, setPopularGames] = useState<Game[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [userId, setUserId] = useState('U000001');
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('discover');

  useEffect(() => {
    loadPopularGames();
    loadGames();
  }, []);

  const loadGames = async () => {
    try {
      const response = await axios.get<Game[]>(`${API_BASE}/games`);
      setGames(response.data);
    } catch (err) {
      console.error('Error loading games:', err);
    }
  };

  const loadPopularGames = async () => {
    try {
      const response = await axios.get<{popular_games: Game[]}>(`${API_BASE}/recommendations/popular?limit=6`);
      setPopularGames(response.data.popular_games);
    } catch (err) {
      console.error('Error loading popular games:', err);
    }
  };

  const getRecommendations = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post<RecommendationResponse>(`${API_BASE}/recommendations/generate`, {
        user_id: userId,
        n_recommendations: 10,
        model_type: 'hybrid'
      });
      
      setRecommendations(response.data.recommendations);
      setActiveTab('recommendations');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to get recommendations');
    } finally {
      setLoading(false);
    }
  };

  const filteredGames = games.filter(game =>
    game.game_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    game.genres.some(genre => genre.toLowerCase().includes(searchTerm.toLowerCase())) ||
    game.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  const renderGameCard = (game: Game, showScore = false) => (
    <Card key={game.game_id} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardMedia
        component="div"
        sx={{
          height: 200,
          background: `linear-gradient(135deg, #667eea 0%, #764ba2 100%)`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'white',
          fontSize: '2rem'
        }}
      >
        <GamepadIcon sx={{ fontSize: '3rem' }} />
      </CardMedia>
      <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
        <Typography gutterBottom variant="h6" component="h3" noWrap>
          {game.game_name}
        </Typography>
        
        <Box sx={{ mb: 1 }}>
          {game.genres.slice(0, 2).map((genre, index) => (
            <Chip
              key={index}
              label={genre}
              size="small"
              sx={{ mr: 0.5, mb: 0.5 }}
              color="primary"
              variant="outlined"
            />
          ))}
        </Box>
        
        <Box sx={{ mb: 1 }}>
          {game.tags.slice(0, 3).map((tag, index) => (
            <Chip
              key={index}
              label={tag}
              size="small"
              sx={{ mr: 0.5, mb: 0.5 }}
              color="secondary"
              variant="outlined"
            />
          ))}
        </Box>
        
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 'auto' }}>
          <Typography variant="h6" color="primary">
            ${game.price.toFixed(2)}
          </Typography>
          {showScore && game.score && (
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <StarIcon sx={{ color: 'gold', mr: 0.5 }} />
              <Typography variant="body2">
                {game.score.toFixed(2)}
              </Typography>
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'discover':
        return (
          <Box>
            <Typography variant="h4" gutterBottom>
              Discover Games
            </Typography>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Search games by name, genre, or tag..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />
              }}
              sx={{ mb: 3 }}
            />
            <Grid container spacing={3}>
              {filteredGames.slice(0, 12).map(game => (
                <Grid item xs={12} sm={6} md={4} lg={3} key={game.game_id}>
                  {renderGameCard(game)}
                </Grid>
              ))}
            </Grid>
          </Box>
        );
      
      case 'recommendations':
        return (
          <Box>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Typography variant="h4" sx={{ mr: 2 }}>
                Your Recommendations
              </Typography>
              <Button
                variant="contained"
                onClick={getRecommendations}
                disabled={loading}
                startIcon={loading ? <CircularProgress size={20} /> : <FavoriteIcon />}
              >
                {loading ? 'Getting Recommendations...' : 'Refresh Recommendations'}
              </Button>
            </Box>
            
            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}
            
            {recommendations.length > 0 ? (
              <Grid container spacing={3}>
                {recommendations.map(game => (
                  <Grid item xs={12} sm={6} md={4} lg={3} key={game.game_id}>
                    {renderGameCard(game, true)}
                  </Grid>
                ))}
              </Grid>
            ) : (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  No recommendations yet
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  Click the button above to get personalized game recommendations!
                </Typography>
              </Box>
            )}
          </Box>
        );
      
      case 'popular':
        return (
          <Box>
            <Typography variant="h4" gutterBottom>
              Popular Games
            </Typography>
            <Grid container spacing={3}>
              {popularGames.map(game => (
                <Grid item xs={12} sm={6} md={4} lg={3} key={game.game_id}>
                  {renderGameCard(game, true)}
                </Grid>
              ))}
            </Grid>
          </Box>
        );
      
      default:
        return null;
    }
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* App Bar */}
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
          <GamepadIcon sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Steam Game Recommender
          </Typography>
          <TextField
            size="small"
            placeholder="User ID"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            sx={{ 
              backgroundColor: 'rgba(255,255,255,0.1)', 
              borderRadius: 1,
              '& .MuiInputBase-input': { color: 'white' },
              '& .MuiInputBase-input::placeholder': { color: 'rgba(255,255,255,0.7)' }
            }}
          />
        </Toolbar>
      </AppBar>

      {/* Navigation Drawer */}
      <Drawer
        anchor="left"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      >
        <Box sx={{ width: 250, pt: 2 }}>
          <List>
            <ListItem button onClick={() => { setActiveTab('discover'); setDrawerOpen(false); }}>
              <ListItemIcon><HomeIcon /></ListItemIcon>
              <ListItemText primary="Discover Games" />
            </ListItem>
            <ListItem button onClick={() => { setActiveTab('recommendations'); setDrawerOpen(false); }}>
              <ListItemIcon><FavoriteIcon /></ListItemIcon>
              <ListItemText primary="Your Recommendations" />
            </ListItem>
            <ListItem button onClick={() => { setActiveTab('popular'); setDrawerOpen(false); }}>
              <ListItemIcon><TrendingIcon /></ListItemIcon>
              <ListItemText primary="Popular Games" />
            </ListItem>
          </List>
        </Box>
      </Drawer>

      {/* Main Content */}
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        {/* Hero Section */}
        <Box sx={{ textAlign: 'center', mb: 6 }}>
          <Typography variant="h2" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
            Discover Your Next Favorite Game
          </Typography>
          <Typography variant="h5" color="text.secondary" paragraph>
            AI-powered recommendations based on your gaming preferences
          </Typography>
          <Button
            variant="contained"
            size="large"
            onClick={getRecommendations}
            disabled={loading}
            startIcon={loading ? <CircularProgress size={24} /> : <FavoriteIcon />}
            sx={{ px: 4, py: 1.5, fontSize: '1.1rem' }}
          >
            {loading ? 'Getting Recommendations...' : 'Get Personalized Recommendations'}
          </Button>
        </Box>

        {/* Tab Content */}
        {renderTabContent()}
      </Container>
    </Box>
  );
} 