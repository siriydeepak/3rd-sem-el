"use client";

import { useGLTF } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useRef, useEffect } from "react";
import * as THREE from "three";

// A custom group that fades its children in/out based on the active prop
function FadeGroup({ active, children, scaleOffset = 1, yOffset = 0 }: { active: boolean, children: React.ReactNode, scaleOffset?: number, yOffset?: number }) {
  const groupRef = useRef<THREE.Group>(null);
  const targetOpacity = active ? 1 : 0;
  const currentOpacity = useRef(active ? 1 : 0);

  useFrame((_, delta) => {
    // Smooth damp opacity
    currentOpacity.current = THREE.MathUtils.damp(currentOpacity.current, targetOpacity, 4, delta);
    
    if (groupRef.current) {
      // Hide if essentially invisible to save rendering
      groupRef.current.visible = currentOpacity.current > 0.01;
      
      // Slight scale effect on reveal
      const s = (0.9 + currentOpacity.current * 0.1) * scaleOffset;
      groupRef.current.scale.setScalar(s);

      // Apply opacity to all materials inside
      groupRef.current.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const mesh = child as THREE.Mesh;
          if (mesh.material) {
            const mat = mesh.material as THREE.Material;
            mat.transparent = true;
            mat.opacity = currentOpacity.current;
            mat.needsUpdate = true;
          }
        }
      });
    }
  });

  return <group ref={groupRef} position={[0, yOffset, 0]}>{children}</group>;
}

export function ModelFile({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  // Clone to avoid mutations if used multiple times, though here it's unique
  const clone = scene.clone();
  return <primitive object={clone} />;
}

export default function MorphingViewer({ step }: { step: number }) {
  const isStep1 = step === 1; // area.glb
  const isStep2 = step === 2; // sample.glb (heatmap)
  const isStep3 = step === 3; // intervention.glb

  // Slow global rotation
  const spinRef = useRef<THREE.Group>(null);
  useFrame((_, delta) => {
    if (spinRef.current) {
      spinRef.current.rotation.y += delta * 0.15;
    }
  });

  return (
    <group ref={spinRef}>
      {/* 3D Modeling Stage */}
      <FadeGroup active={isStep1} scaleOffset={7.0} yOffset={0}>
        <ModelFile url="/area.glb" />
      </FadeGroup>

      {/* Heat Map Stage */}
      <FadeGroup active={isStep2} scaleOffset={7.0} yOffset={0}>
        <ModelFile url="/sample.glb" />
      </FadeGroup>

      {/* Interventions Stage */}
      <FadeGroup active={isStep3} scaleOffset={7.0} yOffset={0}>
        <ModelFile url="/intervention.glb" />
      </FadeGroup>
    </group>
  );
}

// Preload the GLBs for instant switching
useGLTF.preload("/area.glb");
useGLTF.preload("/sample.glb");
useGLTF.preload("/intervention.glb");
