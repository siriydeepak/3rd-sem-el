
"use client";

import { Canvas } from "@react-three/fiber";
import {
  OrbitControls,
  useGLTF,
  Environment,
  ContactShadows,
  Center,
} from "@react-three/drei";
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
    <mesh>
      <boxGeometry args={[0.5, 0.5, 0.5]} />
      <meshStandardMaterial
        color="#06D6A0"
        wireframe
      />
    </mesh>
  );
}

interface GLBViewerProps {
  url: string;
  autoRotate?: boolean;
  height?: string;
}

export default function GLBViewer({
  url,
  autoRotate = true,
  height = "h-80",
}: GLBViewerProps) {
  useEffect(() => {
    useGLTF.preload(url);
  }, [url]);

  return (
    <div
      className={`relative w-full ${height} rounded-lg border border-teal-700 overflow-hidden bg-[#0a1628]`}
    >
      <Canvas
        camera={{ position: [0, 0, 6], fov: 45 }}
        gl={{ antialias: true }}
      >
        <ambientLight intensity={1.4} />
        <directionalLight
          position={[10, 10, 10]}
          intensity={2}
        />
        <directionalLight
          position={[-10, -10, -10]}
          intensity={1.2}
          color="#06D6A0"
        />
        <directionalLight
          position={[0, 10, -10]}
          intensity={0.8}
        />
        <directionalLight
          position={[-10, 10, 10]}
          intensity={0.8}
        />

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
          enableZoom
          enablePan={false}
          minDistance={2}
          maxDistance={20}
        />
      </Canvas>

      <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-2 text-xs font-mono text-teal-400">
        <Loader2 className="h-3 w-3 animate-spin" />
        3D VIEWER
      </div>
    </div>
  );
}
