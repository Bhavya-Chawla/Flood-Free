import axios from "axios";

// Define a type for the expected data structure to ensure type safety
interface WeatherData {
  precipitation: number;
  drainage: number;
  Elevation: number;
  Water_Table: number;
  urbanization: number;
  runoff_coefficient: number;
}

export const getWaterloggingPrediction = async (data: WeatherData) => {
  try {
    // Sending the POST request to the backend API
    const response = await axios.post("http://127.0.0.1:8000/predict", data);

    // Return the response data (prediction)
    return response.data;
  } catch (error) {
    // Improved error handling
    if (axios.isAxiosError(error)) {
      if (error.response) {
        // The server responded with a status code other than 2xx
        console.error("Response error:", error.response.data);
      } else if (error.request) {
        // The request was made but no response was received
        console.error("Request error:", error.request);
      } else {
        // Something else caused the error
        console.error("Error message:", error.message);
      }
    } else {
      console.error("Unknown error:", error);
    }

    // Rethrow error for the calling component to handle
    throw error;
  }
};
