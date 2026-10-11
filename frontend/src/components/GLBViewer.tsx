"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, useGLTF, Environment, ContactShadows, Html, Center } from "@react-three/drei";
import { Suspense, useEffect } from "react";
import { Loader2 } from "lucide-react";

function Model({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  return (
    <Center>
      <primitive object={scene} />
    </Center>
  );
}

function LoadingFallback() {
  return (
    <Html center>
      <div className="flex flex-col items-center gap-2 text-teal-400">
        <Loader2 className="w-8 h-8 animate-spin" />
        <p className="text-xs font-mono">Loading 3D model…</p>
      </div>
    </Html>
  );
}

interface GLBViewerProps {
  url: string;
  autoRotate?: boolean;
  height?: string;
}

export default function GLBViewer({ url, autoRotate = true, height = "h-80" }: GLBViewerProps) {
  // Preload the model
  useEffect(() => {
    useGLTF.preload(url);
  }, [url]);

  return (
    <div className={`w-full ${height} rounded-lg border border-teal-700 overflow-hidden bg-[#0a1628]`}>
      <Canvas
        camera={{ position: [0, 0, 6], fov: 45 }}
        gl={{ antialias: true }}
      >
        {/* Lighting from all sides */}
        <ambientLight intensity={1.4} />
        <directionalLight position={[10, 10, 10]}  intensity={2.0} />
        <directionalLight position={[-10, -10, -10]} intensity={1.2} color="#06D6A0" />
        <directionalLight position={[0, 10, -10]}  intensity={0.8} />
        <directionalLight position={[-10, 10, 10]} intensity={0.8} />

        <Suspense fallback={<LoadingFallback />}>
          <Model url={url} />
          <Environment preset="city" />
          <ContactShadows
            position={[0, -2.5, 0]}
            opacity={0.35}
            scale={12}
            blur={2.5}
            far={5}
          />
        </Suspense>

        <OrbitControls
          autoRotate={autoRotate}
          autoRotateSpeed={1.8}
          enableZoom={true}
          enablePan={false}
          minDistance={2}
          maxDistance={20}
        />
      </Canvas>
    </div>
  );
}
