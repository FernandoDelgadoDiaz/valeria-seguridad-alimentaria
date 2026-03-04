export async function handler(event) {
  try {
    const GITHUB_USER = "FernandoDelgadoDiaz";
    const GITHUB_REPO = "valeria-seguridad-alimentaria";
    const GITHUB_FOLDER = "docs";
    const apiUrl = `https://api.github.com/repos/${GITHUB_USER}/${GITHUB_REPO}/contents/${GITHUB_FOLDER}`;

    const headers = {
      "User-Agent": "INOCUO-App",
      "Accept": "application/vnd.github.v3+json"
    };
    if (process.env.GITHUB_TOKEN) {
      headers["Authorization"] = `token ${process.env.GITHUB_TOKEN}`;
    }

    const response = await fetch(apiUrl, { headers });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Error GitHub API:", response.status, errorText);
      return {
        statusCode: response.status,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ error: `GitHub API error: ${response.status}` })
      };
    }

    const contents = await response.json();

    const esCAAChapter = (nombre) => {
      const n = nombre.toLowerCase();
      return n.includes("capitulo") || n.startsWith("anmat_caa") || n.startsWith("caa_cap");
    };

    const files = contents
      .filter(f => f.type === "file" && f.name.toLowerCase().endsWith(".pdf") && esCAAChapter(f.name))
      .map(f => ({
        name: f.name,
        url: `https://raw.githubusercontent.com/${GITHUB_USER}/${GITHUB_REPO}/main/${GITHUB_FOLDER}/${encodeURIComponent(f.name)}`
      }));

    // Deduplicate: keep only one file per chapter number (prefer longer/newer filename)
    const chapterMap = new Map();
    for (const f of files) {
      const num = extractChapterNumber(f.name);
      if (!chapterMap.has(num)) {
        chapterMap.set(num, f);
      } else {
        // Prefer the file with a longer name (more specific/updated version)
        if (f.name.length > chapterMap.get(num).name.length) {
          chapterMap.set(num, f);
        }
      }
    }
    const deduped = Array.from(chapterMap.values());
    deduped.sort((a, b) => extractChapterNumber(a.name) - extractChapterNumber(b.name));

    return {
      statusCode: 200,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(deduped),
    };

  } catch (err) {
    console.error("Error en list-docs:", err);
    return {
      statusCode: 500,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ error: "Error interno del servidor." })
    };
  }
}

function extractChapterNumber(filename) {
  const fn = filename.toLowerCase();
  // Casos especiales que no siguen el patrón capitulo_X
  if (fn.includes('alimentos_grasos')) return 7;
  const romanMap = [
    ["xxii",22],["xxi",21],["xx",20],["xix",19],["xviii",18],["xvii",17],["xvi",16],["xv",15],
    ["xiv",14],["xiii",13],["xii",12],["xi",11],["ix",9],["viii",8],["vii",7],["vi",6],
    ["iv",4],["iii",3],["ii",2],["x",10],["v",5],["i",1]
  ];
  const capMatch = fn.match(/capitulo[_\s]*/);
  if (capMatch) {
    const rest = fn.slice(capMatch.index + capMatch[0].length);
    for (const [roman, num] of romanMap) {
      if (rest.startsWith(roman)) {
        const nextChar = rest[roman.length];
        if (!nextChar || /[^a-z]/.test(nextChar)) return num;
      }
    }
  }
  const numMatch = fn.match(/capitulo[_\s]*(\d+)/);
  if (numMatch) return parseInt(numMatch[1], 10);
  return 999;
}