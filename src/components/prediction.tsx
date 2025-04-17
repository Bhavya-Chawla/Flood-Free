import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

// TypeScript types for the data that the component will accept
type PredictionChartProps = {
  data: { time: string; prediction: number }[];
};

const PredictionChart: React.FC<PredictionChartProps> = ({ data }) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Waterlogging Prediction</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[160px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data}>
              <XAxis
                dataKey="time"
                stroke="#888888"
                fontSize={12}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                stroke="#888888"
                fontSize={12}
                tickLine={false}
                axisLine={false}
                tickFormatter={(value) => `${value}`} // Show the integer values (0 or 1)
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length > 0) {
                      return (
                        <div className="rounded-lg border bg-background p-2 shadow-sm">
                          <div className="grid grid-cols-2 gap-2">
                            <div className="flex flex-col">
                              <span className="text-[0.70rem] uppercase text-muted-foreground">
                                Waterlogging Prediction Value
                              </span>
                              <span className="font-bold">
                                {payload[0].value + "%"} {/* Shows percentage */}
                              </span>
                            </div>
                          </div>
                        </div>
                      );
                    
                  }
                  return null;
                }}
              />
              <Bar
                dataKey="prediction"
                fill="#2563eb"
                barSize={20} // Smaller bar size
                radius={[5, 5, 0, 0]} // Rounded top corners
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export default PredictionChart;
