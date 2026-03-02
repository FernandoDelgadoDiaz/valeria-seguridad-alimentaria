import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export async function handler(event) {
  console.log("📚 Función list-docs iniciada");

  try {
    // Intentar diferentes rutas posibles para la carpeta docs
    const possiblePaths = [
      path.resolve(process.cwd(), "docs"),                // raíz del proyecto
      path.resolve(__dirname, "..", "..", "docs"),        // desde functions hacia arriba
      path.resolve("/var/task/docs")                       // en entorno Netlify
    ];

    let docsPath = null;
    for (const p of possiblePaths) {
      console.log(`🔍 Verificando ruta: ${p}`);
      if (fs.existsSync(p)) {
        docsPath = p;
        console.log(`✅ Carpeta docs encontrada en: ${docsPath}`);
        break;
      }
    }

    if (!docsPath) {
      console.error("❌ No se encontró la carpeta docs en ninguna ruta");
      return {
        statusCode: 404,
        body: JSON.stringify({ error: "La carpeta docs no existe en el servidor" })
      };
    }

    // Leer archivos de la carpeta docs
    const allFiles = fs.readdirSync(docsPath);
    console.log(`📄 Archivos encontrados en docs: ${allFiles.length}`);

    // Filtrar solo PDFs (o también TXT si querés)
    const files = allFiles
      .filter(f => f.toLowerCase().endsWith(".pdf"))
      .map(f => ({
        name: f,
        url: `/docs/${encodeURIComponent(f)}`
      }));

    console.log(`📚 Archivos PDF encontrados: ${files.length}`);

    if (files.length === 0) {
      console.warn("⚠️ No hay archivos PDF en la carpeta docs");
      return {
        statusCode: 200,
        body: JSON.stringify([])
      };
    }

    // Ordenar por número de capítulo si es posible
    files.sort((a, b) => {
      const numA = extractChapterNumber(a.name);
      const numB = extractChapterNumber(b.name);
      return numA - numB;
    });

    return {
      statusCode: 200,
      headers: { 
        "Content-Type": "application/json",
        "Cache-Control": "no-cache"
      },
      body: JSON.stringify(files),
    };

  } catch (err) {
    console.error("❌ Error en list-docs:", err);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: err.message })
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