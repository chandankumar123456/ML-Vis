import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { Router } from './app/Router';
import { loadAllTopics } from './registry/loadTopics';
import { registerAllViews } from './visualizers/registerViews';
import './styles.css';

// Boot-time topic registration: sidebar, command palette and HomePage all render
// from listTopics(), so modules must be registered before the first paint.
registerAllViews();
loadAllTopics().then(() => {
  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <BrowserRouter>
        <Router />
      </BrowserRouter>
    </React.StrictMode>
  );
});
