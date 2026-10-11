"use client";

import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

export default function Rings() {
  const ring1 = useRef<THREE.Mesh>(null);
  const ring2 = useRef<THREE.Mesh>(null);
  const ring3 = useRef<THREE.Mesh>(null);

  useFrame((state, delta) => {
    if (ring1.current) {
      ring1.current.rotation.x += delta * 0.2;
      ring1.current.rotation.y += delta * 0.1;
    }
    if (ring2.current) {
      ring2.current.rotation.x -= delta * 0.1;
      ring2.current.rotation.y -= delta * 0.2;
    }
    if (ring3.current) {
      ring3.current.rotation.z += delta * 0.05;
      ring3.current.rotation.y += delta * 0.15;
    }
  });

  return (
    <group>
      {/* Outer Cyan Ring */}
      <mesh ref={ring1} rotation={[Math.PI / 3, 0, 0]}>
        <torusGeometry args={[4, 0.02, 16, 100]} />
        <meshBasicMaterial color="#06D6A0" transparent opacity={0.6} />
      </mesh>

      {/* Middle Dark Blue Ring */}
      <mesh ref={ring2} rotation={[Math.PI / 4, Math.PI / 6, 0]}>
        <torusGeometry args={[3.2, 0.015, 16, 100]} />
        <meshBasicMaterial color="#112B4C" transparent opacity={0.8} />
      </mesh>

      {/* Inner Fine Ring */}
      <mesh ref={ring3} rotation={[0, Math.PI / 2, 0]}>
        <torusGeometry args={[2.5, 0.01, 16, 100]} />
        <meshBasicMaterial color="#64ffda" transparent opacity={0.3} />
        
        {/* Floating orbital dots */}
        <mesh position={[2.5, 0, 0]}>
           <sphereGeometry args={[0.08, 16, 16]} />
           <meshBasicMaterial color="#ffffff" />
        </mesh>
        <mesh position={[-2.5, 0, 0]}>
           <sphereGeometry args={[0.05, 16, 16]} />
           <meshBasicMaterial color="#06D6A0" />
        </mesh>
      </mesh>
    </group>
  );
}
