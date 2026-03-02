import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export async function handler(event) {
  console.log("📚 Función list-docs iniciada");

  try {
    // Objeto para depuración
    const debug = {
      possiblePaths: [],
      foundPath: null,
      filesInDocs: [],
      pdfFiles: []
    };

    // Intentar diferentes rutas posibles
    const possiblePaths = [
      path.resolve(process.cwd(), "docs"),
      path.resolve(__dirname, "..", "..", "docs"),
      path.resolve("/var/task/docs")
    ];

    for (const p of possiblePaths) {
      const exists = fs.existsSync(p);
      debug.possiblePaths.push({ path: p, exists });
      if (!debug.foundPath && exists) {
        debug.foundPath = p;
      }
    }

    if (!debug.foundPath) {
      return {
        statusCode: 404,
        body: JSON.stringify({ 
          error: "No se encontró la carpeta docs",
          debug 
        })
      };
    }

    // Leer archivos de la carpeta encontrada
    const allFiles = fs.readdirSync(debug.foundPath);
    debug.filesInDocs = allFiles;

    const pdfFiles = allFiles
      .filter(f => f.toLowerCase().endsWith(".pdf"))
      .map(f => ({
        name: f,
        url: `/docs/${encodeURIComponent(f)}`
      }));

    debug.pdfFiles = pdfFiles.map(f => f.name);

    if (pdfFiles.length === 0) {
      return {
        statusCode: 200,
        body: JSON.stringify({ 
          message: "No hay archivos PDF en docs",
          debug 
        })
      };
    }

    // Ordenar por número de capítulo
    pdfFiles.sort((a, b) => {
      const numA = extractChapterNumber(a.name);
      const numB = extractChapterNumber(b.name);
      return numA - numB;
    });

    return {
      statusCode: 200,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        files: pdfFiles,
        debug 
      }),
    };

  } catch (err) {
    return {
      statusCode: 500,
      body: JSON.stringify({ 
        error: err.message,
        stack: err.stack 
      }),
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