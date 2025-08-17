// frontend/src/services/api.ts
import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
api.interceptors.request.use(
  (config: any) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error: any) => Promise.reject(error)
);

// API methods
export const authApi = {
  login: (username: string, password: string) =>
    api.post('/auth/login', { username, password }),
  
  register: (data: any) =>
    api.post('/auth/register', data),
  
  logout: () => {
    localStorage.removeItem('access_token');
    return Promise.resolve();
  },
};

export const recommendationApi = {
  // Generate personalized recommendations
  generateRecommendations: (data: {
    user_id: string;
    n_recommendations: number;
    model_type: string;
    include_played: boolean;
  }) => api.post('/recommendations/generate', data),
  
  // Get available models and performance metrics
  getModels: () => api.get('/recommendations/models'),
  
  // Get sample users for demo
  getSampleUsers: () => api.get('/recommendations/sample-users'),
  
  // Get system performance metrics
  getMetrics: () => api.get('/recommendations/metrics'),
  
  // Get user recommendation history
  getUserRecommendations: (userId: string, params?: any) =>
    api.get(`/recommendations/user/${userId}`, { params }),
  
  // Submit feedback on recommendations
  submitFeedback: (feedback: any) =>
    api.post('/recommendations/feedback', feedback),
  
  // Record user interaction with a game
  recordInteraction: (interaction: any) =>
    api.post('/recommendations/interaction', interaction),
  
  // Get recommendation explanation
  getExplanation: (recommendationId: string) =>
    api.get(`/recommendations/explanation/${recommendationId}`),
  
  // Refresh user recommendations
  refreshRecommendations: (userId: string) =>
    api.post(`/recommendations/refresh/${userId}`),
  
  // Delete a recommendation
  deleteRecommendation: (recommendationId: string) =>
    api.delete(`/recommendations/${recommendationId}`),
};

export const gameApi = {
  getGame: (gameId: string) =>
    api.get(`/games/${gameId}`),
  
  searchGames: (query: string) =>
    api.get('/games/search', { params: { q: query } }),
  
  getGames: () => api.get('/games'),
};

export default api; 