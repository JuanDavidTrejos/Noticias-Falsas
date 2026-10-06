"""Genera un corpus balanceado de noticias falsas y verdaderas.

Dependencias:
    pip install requests beautifulsoup4

Uso:
    python scrape_corpus.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


# Todas las rutas se calculan a partir de la ubicación del script para que
# funcione igual aunque se ejecute desde otra carpeta.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "corpus"
# Newtral necesita una petición AJAX para cargar más resultados; BBC permite
# recorrer directamente las páginas de su sección temática.
NEWTRAL_INDEX = "https://www.newtral.es/zona-verificacion/fakes/"
NEWTRAL_AJAX = "https://www.newtral.es/wp-admin/admin-ajax.php"
BBC_TOPIC = "https://www.bbc.com/mundo/topics/c7zp57yyz25t"
TARGET_PER_CLASS = 150
REQUEST_DELAY_SECONDS = 0.15

# Una sesión reutiliza la conexión HTTP y conserva las cabeceras comunes.
SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (compatible; PLN-Practica/1.0; "
            "+https://www.newtral.es/)"
        ),
        "Accept-Language": "es-ES,es;q=0.9",
    }
)


@dataclass
class Article:
    """Representa una noticia antes de guardarla en el corpus."""

    label: int
    source_name: str
    url: str
    title: str
    body: str
    date: str = ""
    topic: str = ""

    @property
    def text(self) -> str:
        """Une el título y el cuerpo en el formato exigido por la práctica."""
        return f"{self.title}\n\n{self.body}".strip()


def get(url: str, **kwargs) -> str:
    """Descarga una página y aplica una pausa breve entre peticiones."""
    response = SESSION.get(url, timeout=30, **kwargs)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return response.text


def clean_text(value: str) -> str:
    """Elimina etiquetas HTML y espacios repetidos, sin normalizar el texto."""
    return re.sub(r"\s+", " ", BeautifulSoup(value or "", "html.parser").get_text(" ", strip=True)).strip()


def unique(values: Iterable[str]) -> list[str]:
    """Quita duplicados conservando el orden original de los enlaces."""
    return list(dict.fromkeys(values))


def json_ld_articles(soup: BeautifulSoup) -> Iterable[dict]:
    """Obtiene datos estructurados de tipo JSON-LD incrustados en una página."""
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.string or script.get_text())
        except (json.JSONDecodeError, TypeError):
            continue
        # Algunas páginas contienen una lista y otras un objeto con @graph.
        candidates = data if isinstance(data, list) else data.get("@graph", [data])
        if isinstance(candidates, dict):
            candidates = [candidates]
        for candidate in candidates:
            if isinstance(candidate, dict) and (
                candidate.get("articleBody")
                or candidate.get("headline")
                or candidate.get("name")
            ):
                yield candidate


def parse_article(url: str, source_name: str, label: int) -> Article | None:
    """Extrae una noticia; devuelve None si no tiene contenido suficiente."""
    soup = BeautifulSoup(get(url), "html.parser")
    item = next(json_ld_articles(soup), None)
    title = clean_text(
        str((item or {}).get("headline") or (item or {}).get("name") or "")
    )
    body = clean_text(
        str((item or {}).get("articleBody") or (item or {}).get("description") or "")
    )
    # En ciertas noticias de Newtral el cuerpo no aparece en JSON-LD, por lo
    # que se recupera desde el contenedor HTML propio del sitio.
    if source_name == "newtral" and not body:
        title = title or clean_text(
            (soup.select_one("h1.post-title-1") or soup.select_one("h1")).get_text(" ", strip=True)
            if soup.select_one("h1.post-title-1") or soup.select_one("h1")
            else ""
        )
        content = soup.select_one("section.section-post-content")
        body = clean_text(content.get_text(" ", strip=True) if content else "")
    # BBC a veces publica en JSON-LD solo un resumen. En ese caso, se usan
    # los párrafos visibles dentro de <main> para conservar el artículo.
    if source_name == "bbc_mundo" and len(body.split()) < 100:
        paragraphs = [
            clean_text(paragraph.get_text(" ", strip=True))
            for paragraph in soup.select("main p")
        ]
        paragraphs = [paragraph for paragraph in paragraphs if len(paragraph.split()) >= 8]
        if paragraphs:
            body = " ".join(paragraphs)
        title = title or clean_text(
            (soup.select_one("h1") or soup.select_one("title")).get_text(" ", strip=True)
            if soup.select_one("h1") or soup.select_one("title")
            else ""
        )
    # Se descartan páginas que son vídeos, especiales o errores sin cuerpo.
    if not title or len(body.split()) < 15:
        return None
    return Article(
        label=label,
        source_name=source_name,
        url=url,
        title=title,
        body=body,
        date=str((item or {}).get("datePublished") or (item or {}).get("dateModified") or ""),
        topic="actualidad" if source_name == "bbc_mundo" else "verificacion",
    )


def newtral_nonce() -> str:
    """Solicita el token requerido por el endpoint AJAX de Newtral."""
    response = SESSION.post(
        NEWTRAL_AJAX,
        data={"action": "vog_newtral_es_generate_nonce"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["nonce"]


def collect_newtral_urls(limit: int) -> list[str]:
    """Recorre la paginación AJAX de Newtral y devuelve enlaces de artículos."""
    nonce = newtral_nonce()
    urls: list[str] = []
    for page in range(1, 31):
        # Newtral devuelve el HTML de cada página dentro de una respuesta JSON.
        response = SESSION.post(
            NEWTRAL_AJAX,
            data={
                "action": "vog_newtral_es_verification_list_load_more",
                "page": page,
                "nonce": nonce,
                "post_type": "post",
                "sections_explora": "",
                "taxonomy": "section",
                "person": "",
                "political_party": "",
                "term_taxonomy": "",
                "valoration": "",
                "date_start": "",
                "date_end": "",
                "category_name": "zona-verificacion",
                "initial_load": "yes" if page == 1 else "no",
            },
            timeout=30,
        )
        response.raise_for_status()
        html = response.json().get("html", "")
        soup = BeautifulSoup(html, "html.parser")
        for link in soup.select("a[href]"):
            url = urljoin(NEWTRAL_INDEX, link["href"]).split("#")[0]
            if re.search(r"https://www\.newtral\.es/.+/\d{8}/?$", url):
                urls.append(url.rstrip("/") + "/")
        if len(unique(urls)) >= limit or not response.json().get("more_posts"):
            break
    return unique(urls)[:limit]


def collect_bbc_urls(limit: int) -> list[str]:
    """Recorre páginas de BBC y conserva únicamente URLs de artículos."""
    urls: list[str] = []
    for page in range(1, 16):
        soup = BeautifulSoup(get(f"{BBC_TOPIC}?page={page}"), "html.parser")
        for link in soup.select('a[href*="/mundo/articles/"]'):
            url = urljoin(BBC_TOPIC, link["href"]).split("?")[0]
            if re.fullmatch(r"https://www\.bbc\.com/mundo/articles/[a-z0-9]+", url):
                urls.append(url)
        urls = unique(urls)
        if len(urls) >= limit:
            break
    return urls[:limit]


def scrape(urls: list[str], source_name: str, label: int) -> list[Article]:
    """Descarga y valida una colección de URLs, informando el progreso."""
    articles: list[Article] = []
    for index, url in enumerate(urls, 1):
        try:
            article = parse_article(url, source_name, label)
            if article:
                articles.append(article)
            print(f"\r{source_name}: {index}/{len(urls)}; válidas: {len(articles)}", end="")
        except requests.RequestException as error:
            print(f"\nNo se pudo descargar {url}: {error}")
        except Exception as error:
            print(f"\nNo se pudo procesar {url}: {error}")
    print()
    return articles


def write_corpus(articles: list[Article], output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """Escribe los TXT originales y la tabla de metadatos del corpus."""
    corpus_dir = Path(output_dir)
    false_dir = corpus_dir / "Falso"
    true_dir = corpus_dir / "Verdad"
    false_dir.mkdir(parents=True, exist_ok=True)
    true_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = corpus_dir / "metadata.csv"
    # Cada clase tiene su propia numeración para producir nombres legibles.
    counters = {0: 0, 1: 0}
    rows = []
    for article in articles:
        counters[article.label] += 1
        prefix = "scr_newtral" if article.label == 0 else "scr_bbc_mundo"
        doc_id = f"{prefix}_{counters[article.label]:04d}"
        destination = (false_dir if article.label == 0 else true_dir) / f"{doc_id}.txt"
        # Se conserva el texto natural en UTF-8; la limpieza PLN se hará
        # posteriormente y no modifica estos archivos fuente.
        destination.write_text(article.text, encoding="utf-8")
        rows.append(
            {
                "doc_id": doc_id,
                "label": article.label,
                "source_type": "scraping",
                "source_name": article.source_name,
                "url": article.url,
                "title": article.title,
                "date": article.date,
                "n_words": len(article.text.split()),
                "topic": article.topic,
                "file": f"{'Falso' if article.label == 0 else 'Verdad'}/{doc_id}.txt",
            }
        )
    metadata = pd.DataFrame(rows)
    # Conserva los documentos del dataset y actualiza únicamente los scr_*.
    if metadata_path.exists():
        previous = pd.read_csv(metadata_path)
        previous = previous[
            ~previous["doc_id"].astype(str).str.startswith("scr_")
        ]
        metadata = pd.concat([previous, metadata], ignore_index=True, sort=False)
        metadata = metadata.sort_values("doc_id").reset_index(drop=True)
    metadata.to_csv(metadata_path, index=False, encoding="utf-8")


def main(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_per_class: int = TARGET_PER_CLASS,
) -> None:
    """Coordina la recolección, validación y escritura del corpus balanceado."""
    false_urls = collect_newtral_urls(target_per_class)
    # La sección contiene enlaces a vídeos, especiales y páginas sin cuerpo
    # periodístico; se recolectan candidatos extra para terminar con 150 válidos.
    true_urls = collect_bbc_urls(target_per_class * 2)
    print(f"URLs encontradas: Newtral={len(false_urls)}, BBC={len(true_urls)}")
    false_articles = scrape(false_urls, "newtral", 0)
    true_articles = scrape(true_urls, "bbc_mundo", 1)
    if len(false_articles) < target_per_class or len(true_articles) < target_per_class:
        raise RuntimeError(
            "No se alcanzaron 150 artículos válidos por clase. "
            f"Falsas={len(false_articles)}, verdaderas={len(true_articles)}."
        )
    write_corpus(
        false_articles[:target_per_class] + true_articles[:target_per_class],
        output_dir,
    )
    print(f"Corpus generado correctamente en {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta de salida (por defecto: corpus/).",
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=TARGET_PER_CLASS,
        help="Número de noticias por clase.",
    )
    args = parser.parse_args()
    main(args.output_dir, args.target_per_class)
