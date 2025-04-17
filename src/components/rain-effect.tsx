import React from "react";
import "./RainEffect.css"; // Import the CSS file for the rain effect

interface RainEffectProps {
  title?: string; // Optional title for customization
  className?: string; // Optional className for additional styling
}

const RainEffect: React.FC<RainEffectProps> = ({
  title = "Loading Weather Dashboard",
  className,
}) => {
  return (
    <div className={`rain-effect-container ${className || ""}`}>
      {/* Rain Drops */}
      {Array(12)
        .fill(null)
        .map((_, index) => (
          <div key={index} className="rain"></div>
        ))}

      {/* Title with blinking dots */}
      <h1 className="rain-title">
        {title}
        <span className="dot">.</span>
        <span className="dot">.</span>
        <span className="dot">.</span>
      </h1>
    </div>
  );
};

export default RainEffect;
