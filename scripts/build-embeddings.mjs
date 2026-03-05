import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";
import pdfjsLib from "pdfjs-dist/legacy/build/pdf.js";

// Configuración de PDF.js para entorno Node
pdfjsLib.GlobalWorkerOptions.workerSrc = undefined;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DOCS_DIR = path.resolve(__dirname, "..", "docs");
const DATA_DIR = path.resolve(__dirname, "..", "data");
const OUT_FILE = path.join(DATA_DIR, "embeddings.json");

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

/**
 * Extrae metadata de artículos y capítulo de un fragmento de texto
 */
function extractMetadata(text, filename) {
  // Detectar capítulo desde el nombre del archivo
  const CAP_MAP = {
    'capitulo_i': 'I', 'capitulo_1': 'I', 'disp_grales': 'I',
    'capitulo_ii': 'II', 'establec': 'II',
    'capitulo_iii': 'III', 'prod_alimenticios': 'III',
    'capitulo_iv': 'IV', 'envases': 'IV',
    'capitulo_v': 'V', 'rotulacion': 'V',
    'capitulo_vi': 'VI', 'carneos': 'VI',
    'capitulo_vii': 'VII', 'alimentos_grasos': 'VII', 'caa_cap_alimentos': 'VII',
    'capitulo_viii': 'VIII', 'lacteos': 'VIII',
    'capitulo_ix': 'IX', 'harinas': 'IX',
    'capitulo_x': 'X', 'azucarados': 'X',
    'capitulo_xi': 'XI', 'vegetales': 'XI',
    'capitulo_xii': 'XII', 'aguas': 'XII',
    'capitulo_xiii': 'XIII', 'beb_fermentadas': 'XIII',
    'capitulo_xiv': 'XIV',
    'capitulo_xv': 'XV', 'estimulantes': 'XV',
    'capitulo_xvi': 'XVI', 'correctivos': 'XVI',
    'capitulo_xvii': 'XVII', 'dieteticos': 'XVII',
    'capitulo_xviii': 'XVIII', 'aditivos': 'XVIII',
    'capitulo_xix': 'XIX', 'aislados_prot': 'XIX',
    'capitulo_xx': 'XX', 'metodologia': 'XX',
    'capitulo_xxi': 'XXI', 'procedimientos': 'XXI',
    'capitulo_xxii': 'XXII', 'miscelaneos': 'XXII',
  };

  const fn = filename.toLowerCase();
  let capitulo = null;
  for (const [key, val] of Object.entries(CAP_MAP)) {
    if (fn.includes(key)) { capitulo = val; break; }
  }

  // Detectar artículos mencionados en el texto
  const artRegex = /art[ií]culo[s]?\s*(?:n[°º]?\s*)?(\d+)/gi;
  const articulos = [];
  let match;
  while ((match = artRegex.exec(text)) !== null) {
    const num = parseInt(match[1]);
    if (!articulos.includes(num)) articulos.push(num);
  }

  return { capitulo, articulos };
}

/**
 * Limpia y anonimiza el texto eliminando referencias corporativas
 */
function cleanAndAnonymize(text) {
  return text
    .replace(/La Anónima/gi, "esta organización profesional")
    .replace(/procedimientos de la empresa/gi, "protocolos de buenas prácticas")
    .replace(/estándares de la compañía/gi, "estándares de seguridad alimentaria")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Divide el texto en fragmentos manejables con solapamiento para no perder contexto
 * Aumentamos maxLen a 2500 para reducir la cantidad de fragmentos y el tamaño final.
 */
function chunkText(text, maxLen = 2500, overlap = 300) {
  const chunks = [];
  const sanitized = cleanAndAnonymize(text);
  for (let i = 0; i < sanitized.length; i += (maxLen - overlap)) {
    chunks.push(sanitized.slice(i, i + maxLen));
  }
  return chunks;
}

/**
 * Extrae texto de un PDF página por página
 */
async function extractPdfText(pdfPath) {
  try {
    const u8 = new Uint8Array(await fs.readFile(pdfPath));
    const loadingTask = pdfjsLib.getDocument({ data: u8 });
    const pdf = await loadingTask.promise;

    let out = "";
    for (let p = 1; p <= pdf.numPages; p++) {
      const page = await pdf.getPage(p);
      const content = await page.getTextContent();
      out += content.items.map((i) => i.str).join(" ") + "\n";
    }

    // Si el texto extraído es muy corto, puede ser un PDF escaneado
    if (out.trim().length < 100) {
      console.warn(`⚠️  El PDF ${pdfPath} tiene muy poco texto (${out.length} caracteres). ¿Será una imagen escaneada?`);
    }
    return out;
  } catch (err) {
    console.error(`❌ Error leyendo PDF ${pdfPath}:`, err.message);
    return null; // Indicar que falló
  }
}

async function main() {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("ERROR: OPENAI_API_KEY no detectada en las variables de entorno.");
  }

  await fs.mkdir(DATA_DIR, { recursive: true });

  // Leemos tanto PDFs como archivos de texto
  const files = (await fs.readdir(DOCS_DIR))
    .filter((f) => f.toLowerCase().endsWith(".pdf") || f.toLowerCase().endsWith(".txt"))
    .sort();

  console.log(`📁 Archivos encontrados: ${files.length}`);
  if (files.length === 0) {
    console.warn("⚠️  No hay archivos PDF o TXT en docs/");
    return;
  }

  const finalData = [];
  let totalChunks = 0;

  for (const file of files) {
    console.log(`\n📖 Procesando: ${file}...`);
    const fullPath = path.join(DOCS_DIR, file);
    let rawText;

    if (file.toLowerCase().endsWith(".pdf")) {
      rawText = await extractPdfText(fullPath);
    } else {
      // Archivo de texto: leer directamente
      try {
        rawText = await fs.readFile(fullPath, "utf-8");
      } catch (err) {
        console.error(`❌ Error leyendo archivo de texto ${fullPath}:`, err.message);
        rawText = null;
      }
    }

    if (!rawText) {
      console.log(`⏩ Saltando ${file} por errores de lectura.`);
      continue;
    }

    const parts = chunkText(rawText);
    console.log(`   → ${parts.length} fragmentos generados`);

    if (parts.length === 0) {
      console.log(`⏩ ${file} no generó fragmentos (texto vacío).`);
      continue;
    }

    totalChunks += parts.length;

    // Procesamos en lotes de 100 para optimizar llamadas a la API
    for (let i = 0; i < parts.length; i += 100) {
      const batch = parts.slice(i, i + 100);
      try {
        const resp = await openai.embeddings.create({
          model: "text-embedding-3-small",
          input: batch,
        });

        resp.data.forEach((d, j) => {
          const meta = extractMetadata(batch[j], file);
          finalData.push({
            source: file,
            text: batch[j],
            embedding: d.embedding,
            capitulo: meta.capitulo,
            articulos: meta.articulos,
          });
        });

        console.log(`   → Lote ${Math.floor(i / 100) + 1} procesado (${batch.length} embeddings)`);
      } catch (err) {
        console.error(`   ❌ Error en lote de embeddings:`, err.message);
        // Continuamos con el siguiente lote
      }

      // Breve pausa para respetar límites de la API
      await new Promise((r) => setTimeout(r, 100));
    }
    console.log(`✅ ${file} completado`);
  }

  const payload = {
    version: "inocuo-v1-2026",
    generatedAt: new Date().toISOString(),
    docsCount: files.length,
    chunks: finalData,
  };

  await fs.writeFile(OUT_FILE, JSON.stringify(payload, null, 2));
  console.log(`\n🚀 ¡Cerebro activado! ${finalData.length} vectores guardados en ${OUT_FILE}`);
}

main().catch((e) => {
  console.error("Error fatal en el build:", e);
  process.exit(1);
});