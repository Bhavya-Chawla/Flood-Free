import React from "react";
import "./RainEffect.css";

interface RainEffectProps {
  title?: string; // Optional title for customization
  className?: string; // Optional className for additional styling
}

const RainEffect: React.FC<RainEffectProps> = ({
  title = "Loading Weather Dashboard...",
  className,
}) => {
  const cloudTypes = ["cloud-small", "cloud-medium", "cloud-large"];

  return (
    <div className={`rain-effect-container ${className || ""}`}>
      {/* Clouds */}
      {Array(4)
        .fill(null)
        .map((_, index) => (
          <div
            key={index}
            className={`cloud ${cloudTypes[index % cloudTypes.length]}`}
          ></div>
        ))}

      {/* Rain */}
      {Array(12)
        .fill(null)
        .map((_, index) => (
          <div key={index} className="rain"></div>
        ))}

      {/* Title */}
      <h1 className="rain-title">{title}</h1>
    </div>
  );
};

export default RainEffect;
