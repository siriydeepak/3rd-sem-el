
"use client";

import { useEffect, useState } from "react";
import Scene from "@/components/Scene";
import {
  ArrowRight,
  Leaf,
  Map,
  Box,
  Activity,
  FileText,
  Camera,
} from "lucide-react";
import { motion } from "framer-motion";

const STAGES = [
  {
    id: "01",
    icon: Camera,
    title: "IMAGE PROCESSING",
    sub: "Satellite & drone imagery enhancement",
  },
  {
    id: "02",
    icon: Box,
    title: "3D MODELING",
    sub: "Trellis geometry generation",
  },
  {
    id: "03",
    icon: Map,
    title: "HEAT MAP",
    sub: "Surface temperature overlay",
  },
  {
    id: "04",
    icon: Leaf,
    title: "INTERVENTIONS",
    sub: "Simulating trees & cool pavements",
  },
  {
    id: "05",
    icon: Activity,
    title: "IMPACT SIMULATION",
    sub: "Predicting temperature drops",
  },
  {
    id: "06",
    icon: FileText,
    title: "FINAL REPORT",
    sub: "Data-driven insights & export",
  },
];

type SatelliteStats = {
  source?: string;
  units?: string;
  valid_pixels: number;
  mean_c: number;
  min_c: number;
  max_c: number;
};

const API_BASE =
  process.env.NEXT_PUBLIC_HEATSCAPE_API_URL?.replace(/\/$/, "") ??
  "http://127.0.0.1:8000";

export default function Home() {
  const [step, setStep] = useState(0);
  const [satellite, setSatellite] =
    useState<SatelliteStats | null>(null);
  const [satelliteError, setSatelliteError] = useState("");

  useEffect(() => {
    let cancelled = false;

    fetch(`${API_BASE}/api/satellite/stats`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Could not load satellite data");
        }
        return response.json();
      })
      .then((data: SatelliteStats) => {
        if (!cancelled) setSatellite(data);
      })
      .catch(() => {
        if (!cancelled) {
          setSatelliteError(
            "Satellite statistics unavailable. Check that the HeatScape backend is running on port 8000."
          );
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const activeStage = STAGES[step] ?? STAGES[0];

  return (
    <main className="min-h-screen overflow-x-hidden bg-[var(--background)] font-sans selection:bg-heat-green selection:text-teal-900">
      <nav className="fixed inset-x-0 top-0 z-50 flex items-center justify-between border-b border-teal-700/20 bg-[#071a2b]/95 px-4 py-4 backdrop-blur-md md:px-12 md:py-5">
        <a
          href="/"
          className="flex items-center gap-2"
          aria-label="HeatScape home"
        >
          <Leaf className="h-6 w-6 text-heat-green" />
          <span className="text-xl font-bold uppercase tracking-widest text-white">
            HeatScape
          </span>
        </a>

        <div className="hidden gap-8 text-sm font-semibold uppercase tracking-wider text-gray-400 md:flex">
          <a
            href="#process"
            className="transition-colors hover:text-heat-yellow"
          >
            The Process
          </a>
          <a
            href="#why-mitigate"
            className="transition-colors hover:text-heat-yellow"
          >
            Why Mitigate
          </a>
        </div>
      </nav>

      <section
        id="process"
        className="relative flex min-h-screen flex-col pt-20 md:pt-24"
      >
        <div
          className="pointer-events-none fixed inset-0 z-0 overflow-hidden"
          aria-hidden="true"
        >
          <div className="absolute -left-[10%] -top-[10%] h-[70vw] w-[70vw] rounded-full bg-heat-green opacity-[0.05] blur-[140px]" />
          <div className="absolute -bottom-[20%] -right-[10%] h-[60vw] w-[60vw] rounded-full bg-heat-yellow opacity-[0.05] blur-[160px]" />
        </div>

        {/* Main content: heading and 3D scene */}
        <div className="relative z-10 mx-auto grid w-full max-w-[1440px] flex-1 items-center gap-4 px-5 py-4 sm:px-6 md:px-12 md:py-5 lg:grid-cols-12 lg:gap-6">
          <div className="flex min-w-0 flex-col justify-center lg:col-span-5">
            <motion.div
              key={step}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, ease: "easeOut" }}
            >
              <div className="mb-4 flex items-center gap-3 md:mb-6">
                <span className="h-px w-10 bg-heat-green" />
                <span className="font-mono text-[11px] uppercase tracking-widest text-heat-green">
                  STAGE {activeStage.id} / 06
                </span>
              </div>

              <h1 className="mb-4 text-4xl font-black uppercase leading-[0.95] tracking-tight text-white sm:text-5xl md:mb-6 md:text-7xl">
                {activeStage.title.split(" ").map((word, index) => (
                  <span
                    key={`${word}-${index}`}
                    className="block"
                  >
                    {word}
                  </span>
                ))}
              </h1>

              <p className="mb-5 max-w-md text-base font-light leading-relaxed text-gray-400 md:mb-8 md:text-xl">
                {activeStage.sub}. Watch the environment transform as AI
                analyzes the urban heat morphology.
              </p>

              <a
                href="/pipeline"
                className="group inline-flex w-fit items-center gap-3 rounded-sm bg-heat-green px-5 py-3 font-bold uppercase tracking-widest text-teal-950 transition-colors hover:bg-heat-yellow md:gap-4 md:px-8 md:py-4"
              >
                Try HeatScape
                <ArrowRight className="h-5 w-5 transition-transform group-hover:translate-x-1" />
              </a>
            </motion.div>
          </div>

          {/* 3D scene */}
          <div className="relative flex min-h-[280px] min-w-0 items-center justify-center sm:min-h-[350px] lg:col-span-7 lg:min-h-[440px]">
            <Scene step={step} />
          </div>
        </div>

        {/* Six stage buttons: normal flow, not fixed or absolutely positioned */}
        <div className="relative z-20 mt-auto w-full shrink-0 border-t border-teal-700/30 bg-[#071a2b]/95 backdrop-blur-md">
          <div className="mx-auto grid w-full max-w-[1440px] grid-cols-2 sm:grid-cols-3 lg:grid-cols-6">
            {STAGES.map((stage, index) => {
              const Icon = stage.icon;
              const active = index === step;

              return (
                <button
                  key={stage.id}
                  type="button"
                  onClick={() => setStep(index)}
                  aria-pressed={active}
                  className={`relative flex min-h-[76px] min-w-0 flex-col justify-center gap-2 border-b border-r border-teal-700/20 px-3 py-3 text-left transition-colors sm:min-h-[86px] sm:px-4 lg:border-b-0 ${
                    active
                      ? "bg-teal-800/50 text-heat-green"
                      : "text-gray-400 hover:bg-teal-800/30 hover:text-white"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-[10px] tracking-widest text-gray-500">
                      {stage.id}
                    </span>

                    {active && (
                      <span className="h-2 w-2 shrink-0 rounded-full bg-heat-green" />
                    )}
                  </div>

                  <div className="flex min-w-0 items-start gap-2">
                    <Icon className="mt-0.5 hidden h-4 w-4 shrink-0 sm:block" />
                    <span className="break-words text-[11px] font-bold uppercase leading-snug tracking-wide sm:text-xs">
                      {stage.title}
                    </span>
                  </div>

                  {active && (
                    <motion.div
                      layoutId="active-indicator"
                      className="absolute left-0 top-0 h-[2px] w-full bg-heat-green"
                    />
                  )}
                </button>
              );
            })}
          </div>
        </div>
      </section>

      {/* Satellite data section */}
      <section className="relative z-10 bg-[var(--background)] px-5 py-12 sm:px-6 md:px-12 md:py-20">
        <div className="mx-auto max-w-[1440px]">
          <p className="mb-3 text-sm uppercase tracking-widest text-heat-green">
            Satellite Data • RVCE
          </p>

          <h2 className="mb-4 text-3xl font-black uppercase text-white md:text-5xl">
            Land Surface Temperature
          </h2>

          <p className="mb-10 text-gray-400">
            Temperature statistics calculated from the HeatScape satellite
            raster.
          </p>

          <div className="mb-10 border border-teal-700 bg-teal-900/30 p-4 md:p-6">
            <h3 className="mb-3 text-lg font-bold uppercase tracking-wider text-white">
              RVCE Satellite Heat Map
            </h3>

            <p className="mb-4 text-sm text-gray-400">
              Satellite-derived land surface temperature distribution.
            </p>

            <div className="w-full overflow-hidden rounded-sm bg-black/30">
              <img
                src={`${API_BASE}/api/satellite/heatmap`}
                alt="RVCE satellite land surface temperature heatmap"
                className="block h-auto max-h-[720px] w-full object-contain"
              />
            </div>

            <div
              className="mt-4 h-3 w-full rounded-sm"
              style={{
                background:
                  "linear-gradient(to right, #1e3cde, #00d2dc, #ffe600, #dc1914)",
              }}
            />

            <div className="mt-2 flex justify-between text-xs text-gray-400">
              <span>Cooler pixels</span>
              <span>Hotter pixels</span>
            </div>
          </div>

          {satelliteError ? (
            <p role="status" className="text-red-400">
              {satelliteError}
            </p>
          ) : !satellite ? (
            <p className="text-gray-400">
              Loading satellite statistics…
            </p>
          ) : (
            <>
              <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
                {[
                  {
                    label: "Mean Temperature",
                    value: satellite.mean_c,
                  },
                  {
                    label: "Minimum Temperature",
                    value: satellite.min_c,
                  },
                  {
                    label: "Maximum Temperature",
                    value: satellite.max_c,
                  },
                  {
                    label: "Valid Pixels",
                    value: satellite.valid_pixels,
                  },
                ].map((item) => (
                  <div
                    key={item.label}
                    className="border border-teal-700 bg-teal-900/50 p-6"
                  >
                    <p className="mb-4 text-sm text-gray-400">
                      {item.label}
                    </p>

                    <p className="break-words text-3xl font-bold text-heat-green md:text-4xl">
                      {item.label === "Valid Pixels"
                        ? item.value.toLocaleString()
                        : `${item.value.toFixed(2)} °C`}
                    </p>
                  </div>
                ))}
              </div>

              <p className="mt-6 text-sm text-gray-500">
                Source: {satellite.source ?? "Satellite raster"} · Units:{" "}
                {satellite.units ?? "Celsius"}
              </p>
            </>
          )}
        </div>
      </section>

      {/* Deep dive section */}
      <section
        id="why-mitigate"
        className="relative z-10 bg-teal-900 px-5 py-16 sm:px-6 md:px-12 md:py-28"
      >
        <div className="mx-auto max-w-[1440px]">
          <div className="mb-12 md:mb-16">
            <h2 className="mb-4 text-3xl font-black uppercase tracking-tight text-white md:text-5xl">
              Deep Dive Pipeline
            </h2>

            <p className="max-w-xl text-lg text-gray-300">
              Detailed breakdown of how HeatScape analyzes urban heat
              islands.
            </p>
          </div>

          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3 lg:gap-8">
            {STAGES.map((stage, index) => {
              const Icon = stage.icon;

              return (
                <motion.div
                  key={stage.id}
                  initial={{ opacity: 0, y: 16 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: "-80px" }}
                  transition={{
                    duration: 0.4,
                    delay: index * 0.06,
                  }}
                  className="group border border-teal-700 bg-teal-800 p-6 transition-colors hover:border-heat-green md:p-8"
                >
                  <Icon className="mb-6 h-10 w-10 text-gray-400 transition-colors group-hover:text-heat-green" />

                  <h3 className="mb-3 text-xl font-bold uppercase tracking-wider text-white">
                    {stage.title}
                  </h3>

                  <p className="text-sm leading-relaxed text-gray-300">
                    {stage.sub}. This phase supports the analysis and
                    representation of the urban environment.
                  </p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>
    </main>
  );
}
