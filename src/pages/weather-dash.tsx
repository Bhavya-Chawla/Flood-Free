import { useState, useEffect } from "react";
import { useWeatherQuery, useReverseGeocodeQuery, useForecastQuery } from "@/hooks/use-weather";
import { CurrentWeather } from "../components/current-weather";
import { Alert, AlertDescription, AlertTitle } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { WeatherDetails } from "../components/weather-details";
import { WeatherForecast } from "../components/weather-forecast";
import { HourlyTemperature } from "../components/hourly-temprature";
import { MapPin, AlertTriangle} from "lucide-react";
import { useGeolocation } from "@/hooks/use-geolocation";
import PredictionChart from "../components/prediction";  
import RainEffect from "@/components/rain-effect";


const fetchFiveDayWaterlogging = async (latitude: number, longitude: number) => {
  const apiUrl = "http://localhost:8083/predict"; 
  const forecastData = await fetchFiveDayForecast(latitude, longitude);

  if (!forecastData) return null;

  const waterloggingPredictions = [];

  const currentPrecipitation = await fetchCurrentForecast(latitude, longitude);
  if (currentPrecipitation !== null) {

    const currentInputData = {
        Water_Table: "High",
        urbanization: "Good",
        Elevation:5,
        precipitation: currentPrecipitation, 
        runoff_coefficient: 0.6,
        drainage: 25, 
    };
    
    console.log("Current Data", currentInputData)

    try {
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(currentInputData),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch current waterlogging prediction.");
      }

      const predictionData = await response.json();
        waterloggingPredictions.push(predictionData.waterlogging_probability);
        console.log("Data", waterloggingPredictions)
    } catch (error) {
      console.error("Error in current waterlogging prediction:", error);
      waterloggingPredictions.push(null);
    }
  }

  for (const precip of forecastData) {

    const inputData = {
      Water_Table: "High",
      urbanization: "Good",
      Elevation: 5,
      precipitation: precip,
      runoff_coefficient: 0.6,
      drainage: 25,
    };

    console.log("Input Data", inputData)

    try {
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(inputData),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch waterlogging prediction.");
      }

      const predictionData = await response.json();
      waterloggingPredictions.push(predictionData.waterlogging_probability);
    } catch (error) {
      console.error("Error in waterlogging prediction:", error);
      waterloggingPredictions.push(null);
    }
  }

  return waterloggingPredictions;
};

const fetchFiveDayForecast = async (latitude: number, longitude: number) => {
  const apiKey = import.meta.env.VITE_WEATHERBIT_API_KEY;

  if (!apiKey) {
    console.error("Weatherbit API key is not defined.");
    return null;
  }

  try {
    const url = `https://api.weatherbit.io/v2.0/forecast/daily?lat=${latitude}&lon=${longitude}&key=${apiKey}`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`API request failed with status: ${response.status}`);
    }

    const data = await response.json();
    const forecastData = data.data.map((day: any) => day.precip); 
    return forecastData;
  } catch (error) {
    console.error("Error fetching 5-day forecast:", error);
    return null;
  }
};

const fetchCurrentForecast = async (latitude: number, longitude: number) => {
  const apiKey = import.meta.env.VITE_WEATHERBIT_API_KEY;

  if (!apiKey) {
    console.error("Weatherbit API key is not defined.");
    return null;
  }

  try {
    const url = `https://api.weatherbit.io/v2.0/current?lat=${latitude}&lon=${longitude}&key=${apiKey}`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`API request failed with status: ${response.status}`);
    }

    const data = await response.json();
    const currentPrecipitation = data.data[0].precip;

    return currentPrecipitation;
  } catch (error) {
    console.error("Error fetching current weather data:", error);
    return null;
  }
};

export function WeatherDashboard() {
  const { coordinates, error: locationError, isLoading: locationLoading, getLocation } = useGeolocation();
  const weatherQuery = useWeatherQuery(coordinates);
  const locationQuery = useReverseGeocodeQuery(coordinates);
  const forecastQuery = useForecastQuery(coordinates);

  const [fiveDayWaterlogging, setFiveDayWaterlogging] = useState<number[]>([]);
  const [currentWaterlogging, setCurrentWaterlogging] = useState<number | null>(null); 
  const [showRainEffect, setShowRainEffect] = useState(true);
  const [hideRainEffect, setHideRainEffect] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setHideRainEffect(true); 
      setTimeout(() => setShowRainEffect(false), 10000); 
    }, 1000);

    return () => clearTimeout(timer); 
  }, []);

  useEffect(() => {
    const fetchWaterloggingData = async () => {
      if (coordinates?.lat && coordinates?.lon) {
        const waterloggingData = await fetchFiveDayWaterlogging(coordinates.lat, coordinates.lon);
        setFiveDayWaterlogging(waterloggingData?.slice(1, 6) || []); 
        setCurrentWaterlogging(waterloggingData?.[0] ?? null);
      }
    };

    fetchWaterloggingData();
  }, [coordinates]);

  if (showRainEffect) {
    return <RainEffect className={hideRainEffect ? "hidden" : ""} />;
  }

  if (locationLoading) {
    return;
  }

  if (locationError) {
    return (
      <Alert variant="destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Location Error</AlertTitle>
        <AlertDescription className="flex flex-col gap-4">
          <p>{locationError}</p>
          <Button variant="outline" onClick={getLocation} className="w-fit">
            <MapPin className="mr-2 h-4 w-4" />
            Enable Location
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  if (!coordinates) {
    return (
      <Alert>
        <MapPin className="h-4 w-4" />
        <AlertTitle>Location Required</AlertTitle>
        <AlertDescription className="flex flex-col gap-4">
          <p>Please enable location access to see your local weather.</p>
          <Button variant="outline" onClick={getLocation} className="w-fit">
            <MapPin className="mr-2 h-4 w-4" />
            Enable Location
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  const locationName = locationQuery.data?.[0];

  if (!weatherQuery.data) {
    return;
  }

  if (forecastQuery.isLoading) {
    return;
  }

  if (forecastQuery.isError) {
    return (
      <Alert variant="destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Forecast Error</AlertTitle>
        <AlertDescription>
          Unable to fetch forecast data. Please try again later.
        </AlertDescription>
      </Alert>
    );
  }

  if (!forecastQuery.data) {
    return <p>No forecast data available.</p>;
  }

  const chartData = fiveDayWaterlogging.map((prediction, index) => ({
    time: `Day ${index + 1}`,
    prediction,
  }));

  console.log("Chart data", chartData)

  return (
    <div className="space-y-4">
      
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold tracking-tight">My Location</h1>
      </div>

      <div className="grid gap-6">
        <div className="flex flex-col lg:flex-row gap-4">
          <CurrentWeather data={weatherQuery.data} locationName={locationName} />
          <HourlyTemperature data={forecastQuery.data} />
        </div>

        <div className="grid gap-6 md:grid-cols-2 items-start">
          <WeatherDetails
            data={weatherQuery.data}
            currentWaterlogging={currentWaterlogging} 
          />
          <PredictionChart data={chartData} />
        </div>
        <WeatherForecast data={forecastQuery.data} />
      </div>
    </div>
  );
}

