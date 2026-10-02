from pathlib import Path

PDF_PATH = Path(__file__).with_name("project_summary.pdf")

pages = [
    {
        "title": "Energy-Aware Adaptive Self-Speculative Decoding",
        "lines": [
            "Project summary for understanding and presentation",
            "",
            "Motivation:",
            "- Reduce energy use and latency for small language models on laptops and consumer GPUs.",
            "- Keep output quality stable while avoiding wasted speculative work.",
            "",
            "Research idea:",
            "- Use early-layer draft generation and full-model verification.",
            "- Choose between ordinary decoding and speculative decoding based on energy cost.",
            "- Use recent acceptance history and context length to decide whether a draft is worth it.",
            "",
            "Main research question:",
            "- When does speculative decoding actually save energy per committed token?",
            "- When is ordinary decoding safer and cheaper?",
            "",
            "Current prototype status:",
            "- The repo contains a synthetic controller prototype that chooses draft depth and draft length.",
            "- It simulates energy, latency, and acceptance behavior.",
            "- The prototype passes tests and demonstrates the controller logic.",
            "- The real model version is prepared as the next step.",
        ],
    },
    {
        "title": "What the project does and how it works",
        "lines": [
            "1. Offline preparation",
            "- Profile action candidates such as ordinary decoding, draft depth 4, and draft length 2 or 5.",
            "- Measure acceptance rate, energy, latency, and memory for each configuration.",
            "- Use calibration prompts to fit the controller policy.",
            "",
            "2. Online inference",
            "- At each generation step, the controller checks the current context and acceptance history.",
            "- It compares estimated energy per committed token against ordinary decoding.",
            "- If expected gains are uncertain, it falls back to ordinary decoding.",
            "",
            "3. Hardware support",
            "- The project is designed for CPU, CUDA, Apple MPS, Intel XPU, and simulation mode.",
            "- An RTX 3050 setup script is included for NVIDIA users.",
            "- Real model execution requires PyTorch, Transformers, and CUDA-enabled tooling.",
            "",
            "4. Output and interpretation",
            "- Energy per token, latency, throughput, acceptance rate, action trace, and output matching are reported.",
            "- A good result is lower energy per committed token without changing ordinary output behavior.",
            "",
            "Summary:",
            "This project is a focused research prototype for an energy-aware speculative decoding controller,",
            "starting from a working synthetic simulator and moving toward a real GPU-enabled evaluation pipeline.",
        ],
    },
]


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_page_contents(page_title: str, lines) -> str:
    content = []
    y = 790
    content.append(f"BT /F1 18 Tf 50 {y} Td ({pdf_escape(page_title)}) Tj ET")
    y -= 28
    for line in lines:
        if not line:
            y -= 16
            continue
        if line.startswith("-") or line.startswith("•"):
            content.append(f"BT /F1 11 Tf 70 {y} Td ({pdf_escape(line)}) Tj ET")
        else:
            content.append(f"BT /F1 12 Tf 50 {y} Td ({pdf_escape(line)}) Tj ET")
        y -= 18
    return "\n".join(content)


def build_pdf() -> bytes:
    objects = []
    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    objects.append("<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>")

    page_streams = []
    for page in pages:
        stream_text = build_page_contents(page["title"], page["lines"])
        page_streams.append(stream_text)

    page_objects = []
    for idx, stream_text in enumerate(page_streams, start=1):
        page_object = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            "/Resources << /Font << /F1 5 0 R >> >> /Contents {content_obj} 0 R >>"
        )
        page_objects.append(page_object.format(content_obj=5 + idx))

    objects.extend(page_objects)
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for idx, stream_text in enumerate(page_streams, start=1):
        objects.append(f"<< /Length {len(stream_text.encode('latin-1', 'replace'))} >>\nstream\n{stream_text}\nendstream")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n{obj}\nendobj\n".encode("latin-1", "replace"))

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1"))
    return bytes(pdf)


if __name__ == "__main__":
    PDF_PATH.write_bytes(build_pdf())
    print(f"Created {PDF_PATH}")
