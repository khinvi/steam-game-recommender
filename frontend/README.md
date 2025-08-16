# Steam Game Recommender Frontend

A React-based frontend for the Steam Game Recommender application.

## Features

- Personalized game recommendations
- Trending games
- New releases
- Modern Material-UI design
- Responsive layout

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Build

To build for production:
```bash
npm run build
```

## Dependencies

- React 18
- TypeScript
- Material-UI (MUI)
- Emotion (for styling)

## Project Structure

```
src/
├── components/     # Reusable UI components
├── pages/         # Page components
│   └── index.tsx  # Home page with recommendations
├── services/      # API services
│   └── api.ts     # Recommendation API client
├── App.tsx        # Main app component
└── index.tsx      # App entry point
``` 