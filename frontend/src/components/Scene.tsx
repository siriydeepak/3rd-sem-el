"use client";

import { Canvas } from "@react-three/fiber";
import { Environment, OrbitControls, ContactShadows } from "@react-three/drei";
import MorphingViewer from "./MorphingViewer";
import Rings from "./Rings";
import { Suspense } from "react";
import Image from "next/image";

interface SceneProps {
  step: number;
}

export default function Scene({ step }: SceneProps) {
  // Determine opacity for images based on the active step
  // Step 0: area.png
  // Step 1: area.glb
  // Step 2: sample.glb
  // Step 3: intervention.glb
  // Step 4: impact.png
  // Step 5: report.png

  const getOpacity = (targetStep: number) => step === targetStep ? 1 : 0;
  const show3D = step === 1 || step === 2 || step === 3;

  return (
    <div className="w-full h-full relative flex items-center justify-center">
      
      {/* 3D Canvas layer */}
      <div 
        className="absolute inset-0 z-10 transition-opacity duration-1000 ease-in-out"
        style={{ opacity: show3D ? 1 : 0, pointerEvents: show3D ? "auto" : "none" }}
      >
        <Canvas camera={{ position: [0, 0, 8], fov: 45 }}>
          <ambientLight intensity={1.2} />
          <directionalLight position={[10, 10, 10]} intensity={2.0} />
          <directionalLight position={[-10, -10, -10]} intensity={1.5} color="#06D6A0" />
          <directionalLight position={[0, 10, -10]} intensity={1.0} />
          <directionalLight position={[-10, 10, 10]} intensity={1.0} />
          
          <Suspense fallback={null}>
            <MorphingViewer step={step} />
            {/* Always show rings, they look cool! */}
            <Rings />
            <Environment preset="city" />
            <ContactShadows position={[0, -2.5, 0]} opacity={0.4} scale={10} blur={2} far={4} />
          </Suspense>

          <OrbitControls enableZoom={false} enablePan={false} autoRotate={false} />
        </Canvas>
      </div>

      {/* 2D Image Overlays */}
      <div className="absolute inset-0 z-20 pointer-events-none flex items-center justify-center">
        <div className="relative w-[80%] h-[80%] max-w-2xl max-h-[600px]">
          <Image 
            src="/area.png" 
            alt="Raw Satellite Area" 
            fill 
            className="object-contain transition-opacity duration-1000 ease-in-out drop-shadow-2xl"
            style={{ opacity: getOpacity(0) }} 
          />
          <Image 
            src="/impact.png" 
            alt="Simulated Impact" 
            fill 
            className="object-contain transition-opacity duration-1000 ease-in-out drop-shadow-2xl"
            style={{ opacity: getOpacity(4) }} 
          />
          <Image 
            src="/report.png" 
            alt="Final Report" 
            fill 
            className="object-contain transition-opacity duration-1000 ease-in-out drop-shadow-2xl"
            style={{ opacity: getOpacity(5) }} 
          />
        </div>
      </div>

    </div>
  );
}
