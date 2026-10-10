import "./list-skeleton.css";

export function ListSkeleton({ label, rows = 4 }: { label: string; rows?: number }) {
  return <div className="list-skeleton" role="status" aria-label={label}>
    <span className="visually-hidden">{label}</span>
    <div aria-hidden="true">{Array.from({ length: rows }, (_, index) =>
      <div className="list-skeleton-row" key={index}><i /><span><b /><small /></span><em /></div>
    )}</div>
  </div>;
}
