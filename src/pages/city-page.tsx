import { useState, useEffect } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useWeatherQuery, useForecastQuery } from "@/hooks/use-weather";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { CurrentWeather } from "../components/current-weather";
import { HourlyTemperature } from "../components/hourly-temprature";
import { WeatherDetails } from "../components/weather-details";
import { WeatherForecast } from "../components/weather-forecast";
import PredictionChart from "@/components/prediction";
import * as XLSX from "xlsx"; 

const getValuesFromExcel = async (lat: number, lon: number) => {
  const filePath = "/sid.xlsx"; 
  console.log(lat, lon)
  try {
    const response = await fetch(filePath);
    const arrayBuffer = await response.arrayBuffer();
    const workbook = XLSX.read(arrayBuffer, { type: "array" });
    const sheet = workbook.Sheets[workbook.SheetNames[0]];
    const data = XLSX.utils.sheet_to_json(sheet);

    const roundedLat = parseFloat(lat.toFixed(2));
    const roundedLon = parseFloat(lon.toFixed(2));
    console.log("Excel Data:", data);

    const row = data.find((item: any) =>
        parseFloat(item.Latitude?.toFixed(2)) === roundedLat &&
        parseFloat(item.Longitude?.toFixed(2)) === roundedLon
    );

    if (!row) {
            console.error("No matching data found in Excel file.");
            return null;
        }

        console.log("Matching Row:", row);

    return {
      drainage: row["Drainage"] || null,
      elevation: row["Elevation"] || null,
      waterTable: row["Water Table"] || null,
      urbanization: row["Urbanization"] || null,
      runoffCoefficient: row["Runoff Coefficient"] || null,
    };
  } catch (error) {
    console.error("Error fetching Excel data:", error);
    return null;
  }
};


const fetchWaterloggingPredictions = async (latitude: number, longitude: number) => {
  const apiUrl = "http://localhost:8082/predict"; // API endpoint
  const apiKey = import.meta.env.VITE_WEATHERBIT_API_KEY;

  if (!apiKey) {
    console.error("Weatherbit API key is not defined.");
    return null;
  }

  try {
    const forecastResponse = await fetch(
      `https://api.weatherbit.io/v2.0/forecast/daily?lat=${latitude}&lon=${longitude}&key=${apiKey}`
    );
    if (!forecastResponse.ok) {
        throw new Error(`Forecast API request failed with status: ${forecastResponse.status}`);
      }
    const forecastData = await forecastResponse.json();
    const dailyPrecipitation = forecastData.data.map((day: any) => day.precip);

    const currentWeatherResponse = await fetch(
      `https://api.weatherbit.io/v2.0/current?lat=${latitude}&lon=${longitude}&key=${apiKey}`
    );
    if (!currentWeatherResponse.ok) {
        throw new Error(`Current weather API request failed with status: ${currentWeatherResponse.status}`);
      }
    const currentWeatherData = await currentWeatherResponse.json();
    const currentPrecipitation = currentWeatherData.data[0].precip;

    const predictions = [];

    const excelValues = await getValuesFromExcel(latitude, longitude);
    if (!excelValues) return null;

    const currentInputData = {
        Water_Table: excelValues.waterTable,
        urbanization: excelValues.urbanization,
        Elevation: excelValues.elevation,
        precipitation: currentPrecipitation,
        runoff_coefficient: excelValues.runoffCoefficient,
        drainage: excelValues.drainage,
    };

    const currentPredictionResponse = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentInputData),
    });

    const currentPrediction = await currentPredictionResponse.json();
    predictions.push(currentPrediction.waterlogging_probability);

    console.log(dailyPrecipitation)
    for (const precip of dailyPrecipitation) {
        const inputData = {
            Water_Table: excelValues.waterTable,
            urbanization: excelValues.urbanization,
            Elevation: excelValues.elevation,
            precipitation: precip,
            runoff_coefficient: excelValues.runoffCoefficient,
            drainage: excelValues.drainage,
          };

      const response = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputData),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch waterlogging prediction.");
      }
      const prediction = await response.json();
      predictions.push(prediction.waterlogging_probability);
    }

    console.log("Hey", predictions)
    return predictions;
  } catch (error) {
    console.error("Error fetching waterlogging data:", error);
    return null;
  }
};

export function CityPage() {
  const [searchParams] = useSearchParams();
  const params = useParams();
  const lat = parseFloat(searchParams.get("lat") || "0");
  const lon = parseFloat(searchParams.get("lon") || "0");

  const coordinates = { lat, lon };

  const weatherQuery = useWeatherQuery(coordinates);
  const forecastQuery = useForecastQuery(coordinates);

  const [fiveDayPredictions, setFiveDayPredictions] = useState<number[]>([]);
  const [currentPrediction, setCurrentPrediction] = useState<number | null>(null);


  useEffect(() => {
    const fetchPredictions = async () => {
      const predictions = await fetchWaterloggingPredictions(lat, lon);
      if (predictions) {
        setCurrentPrediction(predictions[0] ?? null);
        setFiveDayPredictions(predictions.slice(1, 6));
      }
    };

    fetchPredictions();
  }, [lat, lon]);

  const chartData = fiveDayPredictions.map((prediction, index) => ({
    time: `Day ${index + 1}`,
    prediction,
  }));

  if (weatherQuery.isLoading || forecastQuery.isLoading) {
    return;
  }

  if (weatherQuery.error || forecastQuery.error) {
    return (
      <Alert variant="destructive">
        <AlertDescription>
          Failed to load weather data. Please try again.
        </AlertDescription>
      </Alert>
    );
  }

  if (!weatherQuery.data || !forecastQuery.data || !params.cityName) {
    return;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
      <h1 className="text-3xl font-bold tracking-tight">
          {params.cityName}, {weatherQuery.data.sys.country}
        </h1>
      </div>


      <div className="grid gap-6">

          <CurrentWeather data={weatherQuery.data} />
          <HourlyTemperature data={forecastQuery.data} />
        </div>

        <div className="grid gap-6 md:grid-cols-2 items-start">
          <WeatherDetails data={weatherQuery.data} currentWaterlogging={currentPrediction} />
          <PredictionChart data={chartData} />
        </div>

        <WeatherForecast data={forecastQuery.data} />
      </div>
  );
}

