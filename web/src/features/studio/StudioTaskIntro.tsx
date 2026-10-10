import chibiArt from "../../assets/themes/nova-chibi.webp";
import "./studio-task-intro.css";

export function StudioTaskIntro({ label, title, description }: { label: string; title: string; description: string }) {
  return <header className="studio-task-intro">
    <div><span>{label}</span><h2>{title}</h2><p>{description}</p></div>
    <img src={chibiArt} alt="" width="88" height="100" />
  </header>;
}
