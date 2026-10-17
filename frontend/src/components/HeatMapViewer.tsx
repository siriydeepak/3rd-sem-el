"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  MapPin,
  Satellite,
  Thermometer,
  Sun,
  Droplets,
  Clock,
  Layers,
  Sparkles,
  ExternalLink,
  Info,
  ShieldAlert,
  Flame,
  TreePine,
} from "lucide-react";

export interface HeatPoint {
  id: string;
  lat: number;
  lon: number;
  distance_m: number;
  is_center: boolean;
  zone_name: string;
  temp_baseline: number;
  temp_mitigated: number;
  temp_drop: number;
  heat_intensity_base: number;
  heat_intensity_mit: number;
  category: string;
}

export interface HeatMapData {
  center: { lat: number; lon: number };
  bounds: { min_lat: number; max_lat: number; min_lon: number; max_lon: number };
  radius_meters: number;
  points: HeatPoint[];
  summary: {
    baseline_avg_temp: number;
    mitigated_avg_temp: number;
    max_hotspot_temp: number;
    min_cool_temp: number;
    total_zones: number;
  };
}

export interface ExifData {
  has_gps: boolean;
  latitude: number;
  longitude: number;
  altitude_m?: number | null;
  timestamp: string;
  camera_make?: string | null;
  camera_model?: string | null;
  source: string;
  notes?: string;
}

export interface WeatherData {
  air_temperature: number;
  satellite_lst: number;
  surface_temperature: number;
  humidity: number;
  solar_exposure: number;
  direct_normal_irradiance: number;
  timestamp_matched: string;
  source: string;
}

export interface HeatAnalysis {
  baseline_lst: number;
  satellite_lst: number;
  air_temperature: number;
  uhi_intensity: number;
  heat_risk: string;
  risk_color: string;
  interventions: {
    cool_pavement: { name: string; predicted_lst: number; temp_reduction_c: number; description: string };
    add_trees: { name: string; predicted_lst: number; temp_reduction_c: number; description: string };
    shade_structure: { name: string; predicted_lst: number; temp_reduction_c: number; description: string };
    combined: { name: string; predicted_lst: number; temp_reduction_c: number; description: string };
  };
}

interface HeatMapViewerProps {
  heatmap: HeatMapData;
  exif?: ExifData;
  weather?: WeatherData;
  heatAnalysis?: HeatAnalysis;
}

// Thermal color helper: cool blue -> teal -> yellow -> orange -> red -> purple
function getHeatColor(temp: number) {
  if (temp < 28) return "bg-sky-500 text-sky-100 border-sky-400";
  if (temp < 32) return "bg-teal-500 text-teal-900 border-teal-400";
  if (temp < 35) return "bg-amber-400 text-amber-950 border-amber-300";
  if (temp < 38) return "bg-orange-500 text-white border-orange-400";
  if (temp < 41) return "bg-red-600 text-white border-red-400";
  return "bg-rose-700 text-white border-rose-500";
}

function getHeatHex(temp: number) {
  if (temp < 28) return "#0284c7"; // sky-600
  if (temp < 32) return "#0d9488"; // teal-600
  if (temp < 35) return "#d97706"; // amber-600
  if (temp < 38) return "#ea580c"; // orange-600
  if (temp < 41) return "#dc2626"; // red-600
  return "#be123c"; // rose-700
}

export default function HeatMapViewer({
  heatmap,
  exif,
  weather,
  heatAnalysis,
}: HeatMapViewerProps) {
  const [showMitigated, setShowMitigated] = useState(false);
  const [selectedPoint, setSelectedPoint] = useState<HeatPoint | null>(
    heatmap.points.find((p) => p.is_center) || heatmap.points[0]
  );

  const centerLat = exif?.latitude ?? heatmap.center.lat;
  const centerLon = exif?.longitude ?? heatmap.center.lon;
  const mapLink = `https://www.google.com/maps?q=${centerLat},${centerLon}`;

  const currentAvg = showMitigated
    ? heatmap.summary.mitigated_avg_temp
    : heatmap.summary.baseline_avg_temp;

  return (
    <div className="w-full bg-[#0a1628]/90 border border-teal-800 rounded-2xl overflow-hidden shadow-2xl backdrop-blur-md">
      {/* ── TOP HEADER: Geotag & Time Badge ── */}
      <div className="p-4 md:p-5 border-b border-teal-800/80 bg-teal-950/40 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-teal-800/60 border border-teal-700 flex items-center justify-center text-heat-green">
            <Satellite className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-white font-bold text-sm tracking-wide">
                Geotagged Thermal Analysis
              </span>
              <span
                className={`text-[10px] px-2 py-0.5 rounded-full font-mono uppercase tracking-wider font-semibold border ${
                  exif?.has_gps
                    ? "bg-green-500/20 text-green-300 border-green-500/40"
                    : "bg-teal-500/20 text-teal-300 border-teal-500/40"
                }`}
              >
                {exif?.has_gps ? "GPS EXIF Verified" : "Campus Location"}
              </span>
            </div>
            <div className="flex items-center gap-3 text-xs text-gray-400 mt-0.5">
              <span className="flex items-center gap-1 font-mono">
                <MapPin className="w-3.5 h-3.5 text-heat-yellow" />
                {centerLat.toFixed(5)}°N, {centerLon.toFixed(5)}°E
              </span>
              {exif?.timestamp && (
                <span className="flex items-center gap-1 font-mono">
                  <Clock className="w-3.5 h-3.5 text-teal-400" />
                  {exif.timestamp}
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <a
            href={mapLink}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs text-teal-300 hover:text-white bg-teal-900/60 hover:bg-teal-800 border border-teal-700 px-3 py-1.5 rounded-lg transition-all"
          >
            <span>Open Map</span>
            <ExternalLink className="w-3 h-3" />
          </a>
        </div>
      </div>

      {/* ── METRIC BAR: SATELLITE VS ML MICROCLIMATE ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-teal-800/40 border-b border-teal-800/80">
        <div className="p-4 bg-[#081220]/80">
          <p className="text-[11px] text-gray-400 flex items-center gap-1">
            <Satellite className="w-3.5 h-3.5 text-sky-400" /> Satellite LST (Macro)
          </p>
          <p className="text-xl md:text-2xl font-black text-sky-300 mt-1">
            {(weather?.satellite_lst ?? heatmap.summary.baseline_avg_temp).toFixed(1)}°C
          </p>
          <p className="text-[10px] text-gray-500 mt-0.5">Landsat / ERA5 assimilation</p>
        </div>

        <div className="p-4 bg-[#081220]/80">
          <p className="text-[11px] text-gray-400 flex items-center gap-1">
            <Thermometer className="w-3.5 h-3.5 text-amber-400" /> Ambient Air Temp
          </p>
          <p className="text-xl md:text-2xl font-black text-amber-300 mt-1">
            {(weather?.air_temperature ?? 31.2).toFixed(1)}°C
          </p>
          <p className="text-[10px] text-gray-500 mt-0.5">2m weather station level</p>
        </div>

        <div className="p-4 bg-[#081220]/80">
          <p className="text-[11px] text-gray-400 flex items-center gap-1">
            <Flame className="w-3.5 h-3.5 text-rose-400" /> Street ML LST (Target)
          </p>
          <p className="text-xl md:text-2xl font-black text-rose-400 mt-1">
            {(heatAnalysis?.baseline_lst ?? heatmap.summary.max_hotspot_temp).toFixed(1)}°C
          </p>
          <p className="text-[10px] text-rose-300 font-semibold mt-0.5">
            {heatAnalysis?.uhi_intensity
              ? `+${heatAnalysis.uhi_intensity}°C UHI Anomaly`
              : "Microclimate Hotspot"}
          </p>
        </div>

        <div className="p-4 bg-[#081220]/80">
          <p className="text-[11px] text-gray-400 flex items-center gap-1">
            <TreePine className="w-3.5 h-3.5 text-heat-green" /> With Interventions
          </p>
          <p className="text-xl md:text-2xl font-black text-heat-green mt-1">
            {(
              (heatAnalysis?.baseline_lst ?? heatmap.summary.max_hotspot_temp) -
              (heatAnalysis?.interventions.combined.temp_reduction_c ?? 3.8)
            ).toFixed(1)}
            °C
          </p>
          <p className="text-[10px] text-heat-green font-semibold mt-0.5">
            -{(heatAnalysis?.interventions.combined.temp_reduction_c ?? 3.8).toFixed(1)}°C Potential Drop
          </p>
        </div>
      </div>

      {/* ── MAP CONTROLS & TOGGLE ── */}
      <div className="p-4 flex flex-wrap items-center justify-between gap-3 bg-teal-950/20 border-b border-teal-800/60">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-gray-300 uppercase tracking-wider">
            Layer Mode:
          </span>
          <div className="flex bg-teal-900/60 border border-teal-700 rounded-lg p-0.5">
            <button
              onClick={() => setShowMitigated(false)}
              className={`px-3 py-1 text-xs rounded-md font-semibold transition-all ${
                !showMitigated
                  ? "bg-rose-600 text-white shadow-md"
                  : "text-gray-400 hover:text-white"
              }`}
            >
              Baseline Heat Map
            </button>
            <button
              onClick={() => setShowMitigated(true)}
              className={`px-3 py-1 text-xs rounded-md font-semibold transition-all ${
                showMitigated
                  ? "bg-heat-green text-teal-950 shadow-md"
                  : "text-gray-400 hover:text-white"
              }`}
            >
              Mitigated (Cooling Plan)
            </button>
          </div>
        </div>

        <div className="flex items-center gap-3 text-xs text-gray-400">
          <span>
            Campus Grid Avg:{" "}
            <strong className="text-white font-mono">{currentAvg.toFixed(1)}°C</strong>
          </span>
          <span className="text-gray-600">|</span>
          <span>
            Radius: <strong className="text-white">{heatmap.radius_meters}m</strong>
          </span>
        </div>
      </div>

      {/* ── INTERACTIVE 5x5 THERMAL SPATIAL GRID ── */}
      <div className="p-5 md:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Spatial Grid Visualizer */}
        <div className="lg:col-span-8 flex flex-col items-center">
          <div className="relative w-full max-w-[480px] aspect-square rounded-2xl p-4 bg-[#040914] border border-teal-700/80 shadow-inner flex flex-col justify-between">
            {/* Compass / Orientation */}
            <div className="absolute top-2 right-3 text-[10px] font-mono text-teal-500 uppercase tracking-widest flex items-center gap-1">
              <span>N ↑</span>
            </div>
            <div className="absolute top-2 left-3 text-[10px] font-mono text-gray-500">
              {heatmap.radius_meters}m Microclimate Grid
            </div>

            {/* Grid Cells (5x5) */}
            <div className="grid grid-cols-5 gap-2 w-full h-full pt-5 pb-3">
              {heatmap.points.map((pt) => {
                const temp = showMitigated ? pt.temp_mitigated : pt.temp_baseline;
                const isSelected = selectedPoint?.id === pt.id;
                const hexColor = getHeatHex(temp);

                return (
                  <motion.button
                    key={pt.id}
                    whileHover={{ scale: 1.06 }}
                    whileTap={{ scale: 0.96 }}
                    onClick={() => setSelectedPoint(pt)}
                    style={{
                      backgroundColor: `${hexColor}22`,
                      borderColor: hexColor,
                    }}
                    className={`relative rounded-xl border flex flex-col items-center justify-center p-1 cursor-pointer transition-all ${
                      isSelected
                        ? "ring-2 ring-white shadow-[0_0_16px_rgba(255,255,255,0.4)] z-10"
                        : "opacity-85 hover:opacity-100"
                    } ${pt.is_center ? "border-dashed" : ""}`}
                  >
                    {pt.is_center && (
                      <span className="absolute -top-2 left-1/2 -translate-x-1/2 bg-yellow-400 text-teal-950 font-black text-[8px] px-1.5 py-0.2 rounded-full uppercase shadow">
                        PHOTO
                      </span>
                    )}
                    <span className="text-[10px] md:text-xs font-mono font-bold text-white">
                      {temp.toFixed(1)}°
                    </span>
                    <span className="text-[8px] text-gray-300 truncate max-w-full px-1">
                      {pt.is_center ? "Photo Site" : `${Math.round(pt.distance_m)}m`}
                    </span>
                  </motion.button>
                );
              })}
            </div>

            {/* Thermal Gradient Legend */}
            <div className="w-full flex items-center justify-between text-[10px] text-gray-400 pt-1 border-t border-teal-900/60">
              <span className="text-sky-300">Cool (&lt;28°C)</span>
              <div className="flex-1 mx-3 h-2 rounded-full bg-gradient-to-r from-sky-500 via-teal-400 via-amber-400 via-orange-500 to-rose-600" />
              <span className="text-rose-400">Extreme (&gt;41°C)</span>
            </div>
          </div>
        </div>

        {/* Right: Selected Zone Details & Intervention Summary */}
        <div className="lg:col-span-4 flex flex-col justify-between space-y-4">
          {selectedPoint && (
            <div className="bg-teal-950/40 border border-teal-700/80 rounded-xl p-4 space-y-3">
              <div className="flex items-start justify-between">
                <div>
                  <span className="text-[10px] font-mono uppercase tracking-wider text-teal-400">
                    Selected Zone Detail
                  </span>
                  <h4 className="text-white font-bold text-sm mt-0.5">
                    {selectedPoint.zone_name}
                  </h4>
                </div>
                <span
                  className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${getHeatColor(
                    showMitigated ? selectedPoint.temp_mitigated : selectedPoint.temp_baseline
                  )}`}
                >
                  {selectedPoint.category}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 pt-1">
                <div className="p-2.5 rounded-lg bg-teal-900/50 border border-teal-800">
                  <p className="text-[10px] text-gray-400">Baseline Surface LST</p>
                  <p className="text-lg font-black text-rose-400">
                    {selectedPoint.temp_baseline.toFixed(1)}°C
                  </p>
                </div>
                <div className="p-2.5 rounded-lg bg-teal-900/50 border border-teal-800">
                  <p className="text-[10px] text-gray-400">Post-Mitigation</p>
                  <p className="text-lg font-black text-heat-green">
                    {selectedPoint.temp_mitigated.toFixed(1)}°C
                  </p>
                </div>
              </div>

              <div className="p-2.5 rounded-lg bg-heat-green/10 border border-heat-green/30 flex items-center justify-between text-xs">
                <span className="text-gray-300 font-medium">Cooling Benefit:</span>
                <span className="text-heat-green font-bold font-mono">
                  -{selectedPoint.temp_drop.toFixed(1)}°C Temperature Drop
                </span>
              </div>

              <p className="text-[11px] text-gray-400 leading-relaxed">
                {selectedPoint.is_center
                  ? "This is the exact site of the ground photo where visual surface features (concrete pavement, brick planters, tree foliage) were analyzed."
                  : `Located ${Math.round(
                      selectedPoint.distance_m
                    )}m from the primary viewpoint. Thermal levels reflect surrounding urban morphology and canopy shading.`}
              </p>
            </div>
          )}

          {/* Quick Intervention Simulators */}
          <div className="bg-teal-900/20 border border-teal-800/80 rounded-xl p-4 space-y-2">
            <p className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-heat-yellow" /> Cooling Interventions
            </p>
            <div className="space-y-1.5 text-xs">
              <div className="flex justify-between items-center text-gray-300 p-1.5 rounded bg-teal-900/40">
                <span>Cool Pavement (High Albedo):</span>
                <span className="text-heat-green font-bold">
                  -{(heatAnalysis?.interventions.cool_pavement.temp_reduction_c ?? 2.1).toFixed(1)}°C
                </span>
              </div>
              <div className="flex justify-between items-center text-gray-300 p-1.5 rounded bg-teal-900/40">
                <span>Shade Trees &amp; Planters:</span>
                <span className="text-heat-green font-bold">
                  -{(heatAnalysis?.interventions.add_trees.temp_reduction_c ?? 2.4).toFixed(1)}°C
                </span>
              </div>
              <div className="flex justify-between items-center text-gray-300 p-1.5 rounded bg-teal-900/40">
                <span>Combined Full Mitigation:</span>
                <span className="text-heat-green font-black">
                  -{(heatAnalysis?.interventions.combined.temp_reduction_c ?? 3.8).toFixed(1)}°C
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
