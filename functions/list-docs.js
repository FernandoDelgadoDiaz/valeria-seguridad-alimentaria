export async function handler(event) {
  console.log("📚 Función list-docs iniciada");

  try {
    const GITHUB_USER = "FernandoDelgadoDiaz";
    const GITHUB_REPO = "valeria-seguridad-alimentaria";
    const GITHUB_FOLDER = "docs";

    const apiUrl = `https://api.github.com/repos/${GITHUB_USER}/${GITHUB_REPO}/contents/${GITHUB_FOLDER}`;

    console.log("🔍 Consultando GitHub API:", apiUrl);

    const response = await fetch(apiUrl, {
      headers: {
        "User-Agent": "INOCUO-App",
        "Accept": "application/vnd.github.v3+json"
      }
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("❌ Error GitHub API:", response.status, errorText);
      return {
        statusCode: response.status,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ error: `GitHub API error: ${response.status}`, detail: errorText })
      };
    }

    const contents = await response.json();

    // Filtrar solo los capítulos del CAA
    const esCAAChapter = (nombre) => {
      const n = nombre.toLowerCase();
      return (
        n.includes("capitulo") ||
        n.startsWith("anmat_caa") ||
        n.startsWith("caa_cap")
      );
    };

    const files = contents
      .filter(f => f.type === "file" && f.name.toLowerCase().endsWith(".pdf") && esCAAChapter(f.name))
      .map(f => ({
        name: f.name,
        url: `https://raw.githubusercontent.com/${GITHUB_USER}/${GITHUB_REPO}/main/${GITHUB_FOLDER}/${encodeURIComponent(f.name)}`
      }));

    console.log(`📚 Capítulos CAA encontrados: ${files.length}`);

    files.sort((a, b) => extractChapterNumber(a.name) - extractChapterNumber(b.name));

    return {
      statusCode: 200,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(files),
    };

  } catch (err) {
    console.error("❌ Error en list-docs:", err);
    return {
      statusCode: 500,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ error: err.message, stack: err.stack })
    };
  }
}

function extractChapterNumber(filename) {
  const match = filename.match(/capitulo[_\s]*([a-z0-9]+)/i);
  if (match) {
    const numStr = match[1].toLowerCase();
    const romanMap = {
      i: 1, ii: 2, iii: 3, iv: 4, v: 5, vi: 6, vii: 7, viii: 8, ix: 9, x: 10,
      xi: 11, xii: 12, xiii: 13, xiv: 14, xv: 15, xvi: 16, xvii: 17, xviii: 18,
      xix: 19, xx: 20, xxi: 21, xxii: 22
    };
    if (romanMap[numStr]) return romanMap[numStr];
    const num = parseInt(numStr, 10);
    if (!isNaN(num)) return num;
  }
  return 999;
}
