# Steam Game Recommender Frontend

This frontend provides two options for the Steam Game Recommender interface:

## Option 1: React/TypeScript Version (Recommended)

The modern React-based frontend with Material-UI components and TypeScript support.

### Features:
- 🎨 Modern Steam-themed design with dark gradient backgrounds
- 📱 Responsive design that works on all devices
- 🎮 Interactive dashboard with performance metrics
- 🔍 Game recommendation system with multiple ML models
- 📊 Statistics and analytics pages
- 🚀 Fast performance with React hooks and state management

### Setup:
```bash
cd frontend
npm install
npm start
```

### Usage:
- **Dashboard**: View system performance metrics and model comparisons
- **Recommendations**: Generate personalized game recommendations
- **Statistics**: View detailed system statistics and dataset information
- **About**: Learn about the technology stack and features

## Option 2: HTML/CSS/JavaScript Version

A standalone HTML version that can be opened directly in any browser.

### Features:
- 🎨 Identical Steam-themed design
- 📱 Fully responsive
- 🚀 No build process required
- 💻 Works offline
- 🎮 Same functionality as React version

### Usage:
Simply open `public/index.html` in your web browser.

## Design Features

### Steam Theme
- **Primary Colors**: Cyan (#00d4ff) and Blue (#66c0f4)
- **Background**: Dark gradient from deep navy to lighter blue
- **Cards**: Glassmorphism effect with backdrop blur
- **Animations**: Smooth hover effects and transitions

### Components
- **Dashboard Cards**: Performance metrics with icons and values
- **Performance Chart**: Interactive bar chart comparing ML models
- **Control Panel**: User-friendly recommendation controls
- **Game Cards**: Beautiful game display with genres and scores
- **Navigation**: Sticky header with smooth page transitions

## API Integration

Both versions are designed to work with the Steam Game Recommender backend API:

- **Base URL**: `http://localhost:8000`
- **Endpoints**: `/games`, `/recommendations/generate`
- **Models**: SVD, Item-Based CF, User-Based CF, Popularity, Hybrid

## Customization

### Colors
Modify the CSS variables in the `:root` selector:
```css
:root {
    --primary: #00d4ff;
    --secondary: #1e3a5f;
    --accent: #66c0f4;
    --dark: #0a0e1a;
    --darker: #050811;
    --light: #c7d5e0;
}
```

### Adding New Pages
1. Create the page content in the appropriate render function
2. Add navigation item to the drawer
3. Update the `renderPageContent` switch statement

## Browser Support

- **Modern Browsers**: Chrome 80+, Firefox 75+, Safari 13+, Edge 80+
- **Features**: CSS Grid, Flexbox, Backdrop Filter, CSS Variables
- **Fallbacks**: Graceful degradation for older browsers

## Performance

- **Lazy Loading**: Components render only when needed
- **Optimized Animations**: CSS transforms and opacity changes
- **Efficient State Management**: React hooks for minimal re-renders
- **Responsive Images**: Optimized for different screen sizes

## Development

### React Version
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server
```

### HTML Version
- Edit `public/index.html` directly
- Refresh browser to see changes
- No build process required

## Deployment

Both versions can be deployed to any static hosting service:
- **Netlify**: Drag and drop the `build` folder or `public` folder
- **Vercel**: Connect your GitHub repository
- **GitHub Pages**: Push to the `gh-pages` branch
- **AWS S3**: Upload static files to S3 bucket

## Troubleshooting

### Common Issues:
1. **Port conflicts**: Change the port in `package.json` scripts
2. **API errors**: Ensure backend is running on `localhost:8000`
3. **Styling issues**: Check browser compatibility for CSS features
4. **Build errors**: Clear `node_modules` and reinstall dependencies

### Getting Help:
- Check the browser console for JavaScript errors
- Verify API endpoints are accessible
- Test with different browsers
- Review the backend API documentation 