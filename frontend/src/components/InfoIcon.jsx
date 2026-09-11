// Small hover-tooltip glossary marker for jargon terms (TMI, IGRF, RTP,
// etc). Uses the native title attribute - no extra state/positioning
// logic needed, and it's keyboard/screen-reader accessible for free.
export default function InfoIcon({ text }) {
  return (
    <span
      title={text}
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: 14,
        height: 14,
        borderRadius: "50%",
        background: "#ddd0b2",
        color: "#4a3d28",
        fontSize: 10,
        fontStyle: "italic",
        fontWeight: 700,
        marginLeft: 4,
        cursor: "help",
        flexShrink: 0,
      }}
    >
      i
    </span>
  );
}
