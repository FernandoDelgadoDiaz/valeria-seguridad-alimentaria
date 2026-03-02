import fs from "fs";
import path from "path";

export async function handler(event) {
  try {
    const docsPath = path.resolve(process.cwd(), "docs");
    const files = fs.readdirSync(docsPath)
      .filter(f => f.toLowerCase().endsWith(".pdf"))
      .map(f => ({
        name: f,
        url: `/docs/${encodeURIComponent(f)}`
      }));

    // Ordenar por número de capítulo si es posible
    files.sort((a, b) => {
      const numA = extractChapterNumber(a.name);
      const numB = extractChapterNumber(b.name);
      return numA - numB;
    });

    return {
      statusCode: 200,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(files),
    };
  } catch (err) {
    return {
      statusCode: 500,
      body: JSON.stringify({ error: err.message }),
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