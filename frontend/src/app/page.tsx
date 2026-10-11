"use client";

import { useState, useEffect } from "react";
import Scene from "@/components/Scene";
import { ArrowRight, Leaf, Map, Box, Activity, FileText, Camera } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const STAGES = [
  { id: "01", icon: Camera, title: "IMAGE PROCESSING", sub: "Satellite & drone imagery enhancement" },
  { id: "02", icon: Box, title: "3D MODELING", sub: "Trellis geometry generation" },
  { id: "03", icon: Map, title: "HEAT MAP", sub: "Surface temperature overlay" },
  { id: "04", icon: Leaf, title: "INTERVENTIONS", sub: "Simulating trees & cool pavements" },
  { id: "05", icon: Activity, title: "IMPACT SIMULATION", sub: "Predicting temperature drops" },
  { id: "06", icon: FileText, title: "FINAL REPORT", sub: "Data-driven insights & export" },
];

export default function Home() {
  const [step, setStep] = useState(1);

  // Auto-advance for presentation, can be removed later
  // useEffect(() => {
  //   const interval = setInterval(() => setStep((s) => (s + 1) % 6), 5000);
  //   return () => clearInterval(interval);
  // }, []);

  const activeStage = STAGES[step];

  return (
    <main className="min-h-screen bg-[var(--background)] overflow-x-hidden font-sans selection:bg-heat-green selection:text-teal-900">
      
      {/* NAVBAR */}
      <nav className="fixed top-0 w-full flex justify-between items-center px-6 py-6 md:px-12 z-50">
        <div className="flex items-center gap-2">
          <Leaf className="text-heat-green w-6 h-6" />
          <span className="text-white font-bold tracking-widest text-xl uppercase">HeatScape</span>
        </div>
        <div className="hidden md:flex gap-8 text-sm font-semibold tracking-wider text-gray-400 uppercase">
          <a href="#" className="hover:text-heat-yellow transition-colors duration-300">The Process</a>
          <a href="#" className="hover:text-heat-yellow transition-colors duration-300">Why Mitigate</a>
        </div>
        <button className="bg-transparent border border-heat-green text-heat-green px-6 py-2 rounded font-semibold hover:bg-heat-green hover:text-teal-900 transition-all duration-300 text-sm tracking-wide uppercase">
          Start Analysis
        </button>
      </nav>

      {/* HERO SECTION - Similar to ReForm3D */}
      <section className="relative w-full min-h-[100svh] flex flex-col pt-24 pb-12">
        
        {/* Background Ambient Glows */}
        <div className="fixed inset-0 pointer-events-none overflow-hidden z-0" aria-hidden="true">
          <div className="absolute top-[-10%] left-[-10%] h-[70vw] w-[70vw] rounded-full bg-heat-green opacity-[0.05] blur-[140px]" />
          <div className="absolute bottom-[-20%] right-[-10%] h-[60vw] w-[60vw] rounded-full bg-heat-yellow opacity-[0.05] blur-[160px]" />
        </div>

        <div className="hero-layout relative z-10 mx-auto grid w-full max-w-[1440px] flex-1 lg:grid-cols-12 px-6 md:px-12">
          
          {/* Text Content (Left side) */}
          <div className="flex flex-col justify-center lg:col-span-5 pt-10 pb-20 lg:py-0">
            <motion.div 
              key={step}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, ease: "easeOut" }}
            >
              <div className="flex items-center gap-3 mb-6">
                <span className="h-px w-10 bg-heat-green" />
                <span className="text-[11px] font-mono tracking-widest text-heat-green uppercase">STAGE {activeStage.id} / {STAGES.length < 10 ? `0${STAGES.length}` : STAGES.length}</span>
              </div>
              
              <h1 className="font-display text-5xl md:text-7xl font-black uppercase leading-[0.95] tracking-tight text-white mb-6">
                {activeStage.title.split(" ").map((word, i) => (
                  <span key={i} className="block">{word}</span>
                ))}
              </h1>
              
              <p className="text-lg md:text-xl font-light text-gray-400 max-w-md leading-relaxed mb-10">
                {activeStage.sub}. Watch the environment transform as AI analyzes the urban heat morphology.
              </p>

              <a
                href="/pipeline"
                className="group flex items-center gap-4 bg-heat-green text-teal-900 px-8 py-4 rounded-sm font-bold tracking-widest uppercase hover:bg-heat-yellow transition-all duration-300 w-fit"
              >
                Try HeatScape
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </a>
            </motion.div>
          </div>

          {/* 3D / 2D Visualizer (Right side) */}
          <div className="relative min-h-[50vh] lg:col-span-7 lg:min-h-full flex items-center justify-center">
            <Scene step={step} />
          </div>
        </div>

        {/* Carousel / Tab selector at bottom */}
        <div className="absolute bottom-0 left-0 w-full z-20 border-t border-teal-700/30 bg-[var(--background)]/80 backdrop-blur-md">
          <div className="flex items-stretch mx-auto max-w-[1440px]">
            {STAGES.map((st, i) => {
              const active = i === step;
              return (
                <button
                  key={st.id}
                  onClick={() => setStep(i)}
                  className={`relative flex-1 flex flex-col items-start p-4 md:p-6 text-left border-r border-teal-700/30 transition-all duration-300 ${
                    active ? "bg-teal-800/50" : "hover:bg-teal-800/30"
                  }`}
                >
                  <div className="flex items-center justify-between w-full mb-2">
                    <span className="text-[10px] font-mono tracking-widest text-gray-500">{st.id}</span>
                    {active && <span className="w-2 h-2 rounded-full bg-heat-green" />}
                  </div>
                  <h3 className={`font-bold text-sm md:text-base uppercase tracking-wider transition-colors ${active ? "text-heat-green" : "text-gray-400"}`}>
                    {st.title}
                  </h3>
                  {active && (
                    <motion.div 
                      layoutId="active-indicator" 
                      className="absolute top-0 left-0 w-full h-[2px] bg-heat-green" 
                    />
                  )}
                </button>
              );
            })}
          </div>
        </div>

      </section>

      {/* MORE SECTIONS BELOW for "scrolling down website and topic cards" */}
      <section className="px-6 py-24 md:px-12 md:py-32 bg-teal-900">
        <div className="max-w-[1440px] mx-auto">
          <div className="mb-16">
            <h2 className="text-3xl md:text-5xl font-black uppercase text-white tracking-tight mb-4">Deep Dive Pipeline</h2>
            <p className="text-gray-400 max-w-xl text-lg">Detailed breakdown of how HeatScape analyzes urban heat islands.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {STAGES.map((st, i) => (
              <motion.div
                key={st.id}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-100px" }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                className="bg-teal-800 p-8 border border-teal-700 hover:border-heat-green transition-colors duration-300 group"
              >
                <st.icon className="w-10 h-10 text-gray-500 group-hover:text-heat-green transition-colors mb-6" />
                <h3 className="text-xl font-bold text-white uppercase tracking-wider mb-3">{st.title}</h3>
                <p className="text-gray-400 text-sm leading-relaxed">{st.sub}. This phase ensures robust data capture and accurate representation for the AI simulator.</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

    </main>
  );
}
