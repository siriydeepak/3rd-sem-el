"use client";

import { useGLTF } from "@react-three/drei";
import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

export default function Model() {
  const { scene } = useGLTF("/sample_model.glb");
  const modelRef = useRef<THREE.Group>(null);

  // Slow rotation for the model
  useFrame((state, delta) => {
    if (modelRef.current) {
      modelRef.current.rotation.y += delta * 0.1;
    }
  });

  return (
    <group ref={modelRef}>
      {/* Center the model loosely and scale it. These might need adjustment based on the actual GLB size */}
      <primitive object={scene} scale={2} position={[0, -1, 0]} />
    </group>
  );
}

// Preload to avoid popping
useGLTF.preload("/sample_model.glb");
