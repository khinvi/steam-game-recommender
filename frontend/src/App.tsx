import React from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import SteamGameRecommender from './components/SteamGameRecommender';

// Create Steam-themed custom theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#00d4ff',
      light: '#66c0f4',
      dark: '#1e3a5f',
    },
    secondary: {
      main: '#66c0f4',
      light: '#8b9df0',
      dark: '#1e3a5f',
    },
    background: {
      default: '#0a0e1a',
      paper: '#1e3a5f',
    },
    text: {
      primary: '#c7d5e0',
      secondary: '#66c0f4',
    },
    success: {
      main: '#5cb85c',
    },
    error: {
      main: '#d9534f',
    },
    warning: {
      main: '#f0ad4e',
    },
  },
  typography: {
    fontFamily: '"Segoe UI", system-ui, -apple-system, sans-serif',
    h1: {
      fontWeight: 700,
      fontSize: '3rem',
    },
    h2: {
      fontWeight: 600,
      fontSize: '2rem',
    },
    h3: {
      fontWeight: 600,
      fontSize: '1.5rem',
    },
    h4: {
      fontWeight: 600,
      fontSize: '1.2rem',
    },
    h5: {
      fontWeight: 500,
      fontSize: '1.1rem',
    },
    h6: {
      fontWeight: 500,
      fontSize: '1rem',
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          background: 'rgba(30, 58, 95, 0.3)',
          backdropFilter: 'blur(10px)',
          borderRadius: 15,
          border: '1px solid rgba(102, 192, 244, 0.2)',
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'translateY(-5px)',
            boxShadow: '0 10px 30px rgba(0, 212, 255, 0.2)',
            borderColor: '#00d4ff',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          background: 'linear-gradient(135deg, #00d4ff, #66c0f4)',
          color: 'white',
          borderRadius: 10,
          textTransform: 'uppercase',
          letterSpacing: '1px',
          fontWeight: 600,
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'translateY(-3px)',
            boxShadow: '0 10px 30px rgba(0, 212, 255, 0.4)',
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            background: 'rgba(10, 14, 26, 0.6)',
            border: '1px solid rgba(102, 192, 244, 0.3)',
            borderRadius: 10,
            color: '#c7d5e0',
            '&:hover': {
              borderColor: '#66c0f4',
            },
            '&.Mui-focused': {
              borderColor: '#00d4ff',
              boxShadow: '0 0 15px rgba(0, 212, 255, 0.3)',
            },
          },
          '& .MuiInputLabel-root': {
            color: '#66c0f4',
          },
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        root: {
          background: 'rgba(10, 14, 26, 0.6)',
          border: '1px solid rgba(102, 192, 244, 0.3)',
          borderRadius: 10,
          color: '#c7d5e0',
          '&:hover': {
            borderColor: '#66c0f4',
          },
          '&.Mui-focused': {
            borderColor: '#00d4ff',
            boxShadow: '0 0 15px rgba(0, 212, 255, 0.3)',
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          background: 'rgba(102, 192, 244, 0.2)',
          border: '1px solid rgba(102, 192, 244, 0.3)',
          color: '#66c0f4',
          borderRadius: 20,
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <SteamGameRecommender />
    </ThemeProvider>
  );
}

export default App; 