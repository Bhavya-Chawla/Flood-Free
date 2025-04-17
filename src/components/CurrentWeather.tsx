interface CurrentWeatherProps {
    data: any;
    locationName: string;
  }
  
  export const CurrentWeather: React.FC<CurrentWeatherProps> = ({ data, locationName }) => {
    if (!data) {
      return <p>Loading current weather...</p>;
    }
  
    const { temp, weather } = data;
    const description = weather?.[0]?.description || "No description available";
  
    return (
      <div className="current-weather">
        <h2>{locationName}</h2>
        <p>{description}</p>
        <p>{temp ? `${temp}°C` : "Temperature not available"}</p>
      </div>
    );
  };
  