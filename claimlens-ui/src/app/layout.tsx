import type { Metadata } from "next";
import { Space_Grotesk, Source_Sans_3 } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import { VerificationProvider } from "@/context/VerificationContext";

const displayFont = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["500", "600", "700"],
});

const bodyFont = Source_Sans_3({
  subsets: ["latin"],
  variable: "--font-body",
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "ClaimLens — AI Fact-Checking Pipeline",
  description:
    "Verify any claim with AI-powered fact-checking. Powered by DeBERTa NLI and LangGraph.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${displayFont.variable} ${bodyFont.variable} font-sans antialiased`}>
        <VerificationProvider>
          <Navbar />
          {children}
          <Footer />
        </VerificationProvider>
      </body>
    </html>
  );
}
