export function KidIllustration({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 240 220"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="A small drawing of a kid reading Shakespeare with a thought bubble showing ROMEO."
    >
      {/* thought bubble */}
      <ellipse
        cx="180"
        cy="42"
        rx="44"
        ry="26"
        fill="#eef2ff"
        stroke="#1f2937"
        strokeWidth="1.75"
      />
      <circle
        cx="138"
        cy="74"
        r="5"
        fill="#eef2ff"
        stroke="#1f2937"
        strokeWidth="1.75"
      />
      <circle
        cx="128"
        cy="86"
        r="3"
        fill="#eef2ff"
        stroke="#1f2937"
        strokeWidth="1.75"
      />
      <text
        x="180"
        y="46"
        textAnchor="middle"
        fontSize="13"
        fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
        fontWeight="700"
        fill="#1f2937"
      >
        ROMEO:
      </text>

      {/* hair tuft */}
      <path
        d="M62 76 Q68 50 92 56 Q104 48 118 60"
        stroke="#1f2937"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* head */}
      <circle
        cx="90"
        cy="95"
        r="32"
        fill="#fff7ed"
        stroke="#1f2937"
        strokeWidth="2"
      />

      {/* eyes */}
      <circle cx="80" cy="93" r="2.2" fill="#1f2937" />
      <circle cx="100" cy="93" r="2.2" fill="#1f2937" />

      {/* mouth — small, curious smile */}
      <path
        d="M82 108 Q90 114 98 108"
        stroke="#1f2937"
        strokeWidth="2"
        strokeLinecap="round"
        fill="none"
      />

      {/* cheek */}
      <circle cx="74" cy="103" r="2.5" fill="#fda4af" opacity="0.55" />
      <circle cx="106" cy="103" r="2.5" fill="#fda4af" opacity="0.55" />

      {/* body — sweater */}
      <path
        d="M68 124 Q60 152 70 178 L132 178 Q142 152 134 124 Q112 134 90 134 Q78 134 68 124 Z"
        fill="#bfdbfe"
        stroke="#1f2937"
        strokeWidth="2"
        strokeLinejoin="round"
      />

      {/* arms holding the book */}
      <path
        d="M70 138 Q56 158 70 172"
        stroke="#1f2937"
        strokeWidth="2"
        strokeLinecap="round"
        fill="none"
      />
      <path
        d="M132 138 Q146 158 132 172"
        stroke="#1f2937"
        strokeWidth="2"
        strokeLinecap="round"
        fill="none"
      />

      {/* book — open, on lap */}
      <path
        d="M44 178 Q72 170 100 178 Q128 170 156 178 L156 206 Q128 198 100 206 Q72 198 44 206 Z"
        fill="#fde68a"
        stroke="#1f2937"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <line
        x1="100"
        y1="178"
        x2="100"
        y2="206"
        stroke="#1f2937"
        strokeWidth="1.5"
      />
      {/* lines of "text" on each page */}
      <line x1="54" y1="186" x2="92" y2="186" stroke="#1f2937" strokeWidth="1" opacity="0.7" />
      <line x1="54" y1="192" x2="92" y2="192" stroke="#1f2937" strokeWidth="1" opacity="0.7" />
      <line x1="54" y1="198" x2="92" y2="198" stroke="#1f2937" strokeWidth="1" opacity="0.7" />
      <line x1="108" y1="186" x2="146" y2="186" stroke="#1f2937" strokeWidth="1" opacity="0.7" />
      <line x1="108" y1="192" x2="146" y2="192" stroke="#1f2937" strokeWidth="1" opacity="0.7" />
      <line x1="108" y1="198" x2="146" y2="198" stroke="#1f2937" strokeWidth="1" opacity="0.7" />
    </svg>
  );
}
