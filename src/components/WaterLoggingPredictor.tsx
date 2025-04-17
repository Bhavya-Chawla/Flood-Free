import { getWaterloggingPrediction } from '@/api/api';
import React, { useState } from 'react';
 // Import the function from api.ts

const WaterLoggingPredictor: React.FC = () => {
  const [prediction, setPrediction] = useState<number | null>(null);
  const [formData, setFormData] = useState({
    precipitation: 0,
    drainage: 0,
    Elevation: 0,
    Water_Table: 0,
    urbanization: 0,
    runoff_coefficient: 0,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const predictionData = await getWaterloggingPrediction(formData);
      setPrediction(predictionData.prediction); // Assuming the backend sends the prediction in `prediction`
    } catch (error) {
      console.error('Error in prediction:', error);
    }
  };

  return (
    <div className="max-w-lg mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Waterlogging Prediction</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        <input
          type="number"
          className="input input-bordered w-full"
          placeholder="Precipitation"
          value={formData.precipitation}
          onChange={(e) => setFormData({ ...formData, precipitation: +e.target.value })}
        />
        <input
          type="number"
          className="input input-bordered w-full"
          placeholder="Drainage"
          value={formData.drainage}
          onChange={(e) => setFormData({ ...formData, drainage: +e.target.value })}
        />
        {/* Add input fields for other parameters */}
        <button type="submit" className="btn btn-primary w-full">Get Prediction</button>
      </form>
      {prediction !== null && <p className="mt-4">Predicted Waterlogging: {prediction}</p>}
    </div>
  );
};

export default WaterLoggingPredictor;
