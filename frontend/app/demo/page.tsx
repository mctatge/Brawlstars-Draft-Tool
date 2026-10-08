import type { Metadata } from "next";
import DocNav from "@/components/DocNav";
import { DocFooter } from "@/components/ContentPage";
import DemoDraft from "@/components/DemoDraft";

export const metadata: Metadata = { title: "Try an example draft — Brawl Draft", description: "Explore a recorded draft without an account or a live API connection.", alternates: { canonical: "/demo" } };

export default function Page() {
  return <main className="min-h-screen p-3 md:p-5 max-w-5xl mx-auto"><DocNav current="/demo" /><DemoDraft /><DocFooter current="/demo" /></main>;
}
