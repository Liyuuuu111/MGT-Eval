import React from 'react';
import ReactDOM from 'react-dom/client';
import { ConfigProvider } from 'antd';
import App from './App';
import 'antd/dist/reset.css';

const theme = {
  token: {
    colorPrimary: '#7c3aed',
    colorSuccess: '#10b981',
    colorError: '#ef4444',
    colorWarning: '#f59e0b',
    colorInfo: '#6366f1',
    borderRadius: 8,
    colorBgLayout: '#f5f3ff',
  },
  components: {
    Card: { borderRadiusLG: 12 },
    Button: { borderRadius: 8 },
    Menu: {
      itemSelectedBg: '#ede9fe',
      itemSelectedColor: '#7c3aed',
      itemHoverBg: '#f5f3ff',
    },
  },
};

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ConfigProvider theme={theme}>
      <App />
    </ConfigProvider>
  </React.StrictMode>
);
