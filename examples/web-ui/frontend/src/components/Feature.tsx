import type { ReactNode } from "react";

/** Consistent page chrome: title, one-line blurb, a "How it works" panel, and
 * the live demo. */
export function Feature({
  title,
  blurb,
  how,
  children,
}: {
  title: string;
  blurb: string;
  how: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="feature">
      <h1>{title}</h1>
      <p className="blurb">{blurb}</p>
      <details className="how" open>
        <summary>How it works</summary>
        <div className="howbody">{how}</div>
      </details>
      <div className="demo">{children}</div>
    </div>
  );
}
