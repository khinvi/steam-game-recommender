// frontend/src/services/api.ts
import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

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
  getPersonalized: (n: number = 10, modelType: string = 'svd') =>
    api.get('/recommendations/personalized', { params: { n, model_type: modelType } }),
  
  getSimilar: (gameId: string, n: number = 10) =>
    api.get(`/recommendations/similar/${gameId}`, { params: { n } }),
  
  getTrending: (period: string = 'week', n: number = 10) =>
    api.get('/recommendations/trending', { params: { period, n } }),
};

export const gameApi = {
  getGame: (gameId: string) =>
    api.get(`/games/${gameId}`),
  
  searchGames: (query: string) =>
    api.get('/games/search', { params: { q: query } }),
};

export default api; 