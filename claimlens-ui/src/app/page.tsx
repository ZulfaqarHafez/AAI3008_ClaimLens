import Hero from "@/components/landing/Hero";
import HowItWorks from "@/components/landing/HowItWorks";
import Architecture from "@/components/landing/Architecture";
import Features from "@/components/landing/Features";
import CTA from "@/components/landing/CTA";

export default function Home() {
  return (
    <main>
      <Hero />
      <HowItWorks />
      <Architecture />
      <Features />
      <CTA />
    </main>
  );
}
