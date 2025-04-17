// In App.tsx
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import './App.css';
import Layout from './components/layout';
import { ThemeProvider } from './context/theme-provider';
import { WeatherDashboard } from "./pages/weather-dash";
import { CityPage } from "./pages/city-page1";
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes
      retry: false,
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <ThemeProvider defaultTheme='dark'>
          {/* Background Image Here */}
          <div className="bg-[url('/image.png')] dark:bg-[url('/dark-image.png')] bg-cover bg-center h-screen">
            <Layout>
              <Routes>
                <Route path='/' element={<WeatherDashboard />} />
                <Route path='/city/:cityName' element={<CityPage />} />
              </Routes>
            </Layout>
          </div>
        </ThemeProvider>
      </BrowserRouter>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

export default App;
