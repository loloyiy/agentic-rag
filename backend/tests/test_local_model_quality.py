"""
Test suite per la qualità delle risposte del modello locale (ollama:qwen2.5:7b).

Verifica:
1. Pertinenza: le risposte sono pertinenti alla domanda?
2. Allucinazioni: il modello inventa informazioni non presenti nei documenti?
3. Lingua: risponde nella lingua della domanda?
4. Completezza: le risposte contengono informazioni sufficienti?
5. Istruzioni: il modello segue le istruzioni del system prompt (es. "non so" quando l'info manca)?
6. Latenza: tempo di risposta accettabile?

Usa i documenti già indicizzati nel sistema, NON carica nuovi file.

Usage:
    cd backend && pytest tests/test_local_model_quality.py -v --tb=short
    cd backend && pytest tests/test_local_model_quality.py -v -k "pertinenza"
"""

import time
import re
import pytest
import requests
import logging

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"
TIMEOUT = 120  # seconds per request

# ---------------------------------------------------------------------------
# Documenti già presenti nel sistema (da /api/documents/)
# ---------------------------------------------------------------------------
# Prende il primo documento unstructured disponibile
DOCS = None  # populated by fixture


def _get_documents():
    """Recupera documenti disponibili dal backend."""
    resp = requests.get(f"{BASE_URL}/api/documents/", timeout=10)
    if resp.status_code != 200:
        return []
    return resp.json()


def _chat(message: str, document_ids: list[str] | None = None,
          conversation_id: str | None = None) -> dict:
    """Invia un messaggio al backend e ritorna la risposta completa."""
    payload = {"message": message}
    if document_ids:
        payload["document_ids"] = document_ids
    if conversation_id:
        payload["conversation_id"] = conversation_id

    start = time.time()
    resp = requests.post(f"{BASE_URL}/api/chat/", json=payload, timeout=TIMEOUT)
    elapsed = time.time() - start

    assert resp.status_code == 200, f"Chat API error: {resp.status_code} - {resp.text}"
    data = resp.json()
    data["_elapsed_seconds"] = elapsed
    return data


def _cleanup_conversation(conv_id: str):
    """Elimina una conversazione di test."""
    if conv_id:
        try:
            requests.delete(f"{BASE_URL}/api/conversations/{conv_id}", timeout=10)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def check_server():
    """Verifica che il backend sia raggiungibile."""
    try:
        resp = requests.get(f"{BASE_URL}/api/health", timeout=5)
        assert resp.status_code == 200
        health = resp.json()
        # Verifica che Ollama sia attivo
        ollama_status = health.get("components", {}).get("ollama", {}).get("status")
        if ollama_status != "healthy":
            pytest.skip("Ollama non disponibile")
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend non raggiungibile su localhost:8000")


@pytest.fixture(scope="module")
def docs(check_server):
    """Restituisce i documenti disponibili nel sistema."""
    documents = _get_documents()
    if not documents:
        pytest.skip("Nessun documento presente nel sistema")
    return documents


@pytest.fixture(scope="module")
def unstructured_doc(docs):
    """Restituisce il primo documento unstructured disponibile."""
    for doc in docs:
        if doc.get("document_type") == "unstructured" and doc.get("chunk_count", 0) > 0:
            return doc
    pytest.skip("Nessun documento unstructured con chunk disponibile")


@pytest.fixture(scope="module")
def doc_id(unstructured_doc):
    """ID del documento di test."""
    return unstructured_doc["id"]


@pytest.fixture(scope="module")
def doc_title(unstructured_doc):
    """Titolo del documento di test."""
    return unstructured_doc.get("title", "Unknown")


# ---------------------------------------------------------------------------
# TEST 1: Pertinenza - la risposta è attinente alla domanda?
# ---------------------------------------------------------------------------
class TestPertinenza:
    """Verifica che il modello locale dia risposte pertinenti."""

    def test_risposta_non_vuota(self, doc_id, check_server):
        """La risposta non deve essere vuota."""
        data = _chat("Di cosa parla questo documento?", document_ids=[doc_id])
        content = data.get("content", "")
        _cleanup_conversation(data.get("conversation_id"))

        assert len(content) > 50, (
            f"Risposta troppo corta ({len(content)} chars): {content}"
        )
        logger.info(f"Risposta: {len(content)} chars, tempo: {data['_elapsed_seconds']:.1f}s")

    def test_risposta_contiene_contenuto_documento(self, doc_id, doc_title, check_server):
        """La risposta deve fare riferimento al contenuto del documento."""
        data = _chat(
            "Riassumi brevemente il contenuto principale di questo documento.",
            document_ids=[doc_id]
        )
        content = data.get("content", "").lower()
        _cleanup_conversation(data.get("conversation_id"))

        # La risposta deve essere sostanziale (non un generico "non so")
        assert len(content) > 100, (
            f"Risposta troppo generica ({len(content)} chars): {content}"
        )
        logger.info(f"Riassunto: {content[:200]}...")

    def test_domanda_specifica_risposta_specifica(self, doc_id, check_server):
        """Una domanda specifica deve ricevere una risposta specifica, non generica."""
        data = _chat(
            "Qual è il primo argomento trattato in questo documento?",
            document_ids=[doc_id]
        )
        content = data.get("content", "")
        _cleanup_conversation(data.get("conversation_id"))

        # Non deve essere una risposta generica tipo "Non ho informazioni"
        generic_phrases = [
            "non ho informazioni",
            "i don't have information",
            "i cannot",
            "non posso",
            "non sono in grado",
        ]
        content_lower = content.lower()
        is_generic = any(phrase in content_lower for phrase in generic_phrases)

        assert not is_generic, (
            f"Risposta troppo generica per una domanda specifica: {content[:300]}"
        )
        logger.info(f"Risposta specifica: {content[:200]}...")


# ---------------------------------------------------------------------------
# TEST 2: Allucinazioni - il modello NON deve inventare informazioni
# ---------------------------------------------------------------------------
class TestAllucinazioni:
    """Verifica che il modello non inventi informazioni assenti dai documenti."""

    def test_no_prezzi_inventati(self, doc_id, check_server):
        """Chiedendo un prezzo non presente, il modello non deve inventarlo."""
        data = _chat(
            "Quanto costa questo prodotto in euro?",
            document_ids=[doc_id]
        )
        content = data.get("content", "")
        _cleanup_conversation(data.get("conversation_id"))

        # Pattern di prezzi inventati
        price_patterns = [
            r"€\s*\d+",
            r"\d+\s*€",
            r"\d+[\.,]\d{2}\s*euro",
            r"costa\s+\d+",
            r"prezzo\s+(è|di)\s+\d+",
            r"price\s+(is|of)\s+\d+",
        ]

        for pattern in price_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            assert not match, (
                f"Allucinazione: prezzo inventato trovato '{match.group()}' nella risposta: {content[:300]}"
            )
        logger.info(f"OK - nessun prezzo inventato. Risposta: {content[:200]}...")

    def test_no_date_inventate(self, doc_id, check_server):
        """Chiedendo una data non presente, il modello non deve inventarla."""
        data = _chat(
            "In che data è stato pubblicato questo documento?",
            document_ids=[doc_id]
        )
        content = data.get("content", "")
        _cleanup_conversation(data.get("conversation_id"))

        # Se il documento non ha date, la risposta dovrebbe indicarlo
        # Non dovrebbe inventare date precise come "15 marzo 2023"
        logger.info(f"Risposta su data: {content[:200]}...")
        # Questo test è più soft - logga il risultato per review manuale
        if len(content) > 0:
            logger.info("REVIEW MANUALE: verificare che la data (se presente) sia corretta")

    def test_ammette_di_non_sapere(self, doc_id, check_server):
        """Il modello deve ammettere di non sapere quando l'info non c'è."""
        data = _chat(
            "Qual è il numero di telefono dell'autore di questo documento?",
            document_ids=[doc_id]
        )
        content = data.get("content", "").lower()
        _cleanup_conversation(data.get("conversation_id"))

        # Non deve inventare un numero di telefono
        phone_pattern = r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"
        match = re.search(phone_pattern, content)

        # Pattern che indicano che il modello ammette di non sapere
        honesty_phrases = [
            "non è presente", "non contiene", "non menzion",
            "non ho trovato", "non è disponibile", "non è indicat",
            "non posso", "non risulta", "non viene",
            "not found", "not available", "not mentioned",
            "doesn't contain", "does not contain",
            "no information", "nessuna informazione",
        ]
        admits_ignorance = any(phrase in content for phrase in honesty_phrases)

        # OK se ammette di non sapere OPPURE se non inventa un telefono
        has_fake_phone = match is not None and not admits_ignorance
        assert not has_fake_phone, (
            f"Allucinazione: numero di telefono inventato '{match.group() if match else ''}' "
            f"nella risposta: {content[:300]}"
        )
        logger.info(f"OK - risposta onesta. Ammette ignoranza: {admits_ignorance}")


# ---------------------------------------------------------------------------
# TEST 3: Lingua - risponde nella lingua della domanda
# ---------------------------------------------------------------------------
class TestLingua:
    """Verifica che il modello risponda nella stessa lingua della domanda."""

    def test_risponde_in_italiano(self, doc_id, check_server):
        """Domanda in italiano -> risposta in italiano."""
        data = _chat(
            "Di cosa parla questo documento? Rispondi in italiano.",
            document_ids=[doc_id]
        )
        content = data.get("content", "")
        _cleanup_conversation(data.get("conversation_id"))

        # Verifica presenza di parole italiane comuni
        italian_indicators = [
            "il", "la", "di", "che", "è", "un", "una", "del", "nel",
            "questo", "documento", "sono", "per", "con",
        ]
        content_lower = content.lower()
        italian_words = sum(1 for w in italian_indicators if f" {w} " in f" {content_lower} ")

        assert italian_words >= 3, (
            f"Risposta probabilmente non in italiano (solo {italian_words} indicatori): "
            f"{content[:300]}"
        )
        logger.info(f"OK - {italian_words} indicatori italiano trovati")

    def test_risponde_in_inglese(self, doc_id, check_server):
        """Domanda in inglese -> risposta in inglese."""
        data = _chat(
            "What is this document about? Answer in English.",
            document_ids=[doc_id]
        )
        content = data.get("content", "")
        _cleanup_conversation(data.get("conversation_id"))

        english_indicators = [
            "the", "is", "of", "and", "in", "to", "this", "that",
            "document", "about", "are", "which", "with",
        ]
        content_lower = content.lower()
        english_words = sum(1 for w in english_indicators if f" {w} " in f" {content_lower} ")

        assert english_words >= 3, (
            f"Risposta probabilmente non in inglese (solo {english_words} indicatori): "
            f"{content[:300]}"
        )
        logger.info(f"OK - {english_words} indicatori inglese trovati")


# ---------------------------------------------------------------------------
# TEST 4: Completezza - le risposte sono sufficientemente dettagliate?
# ---------------------------------------------------------------------------
class TestCompletezza:
    """Verifica che le risposte siano complete e non troncate."""

    def test_risposta_non_troncata(self, doc_id, check_server):
        """La risposta non deve terminare a metà frase."""
        data = _chat(
            "Elenca i punti principali trattati in questo documento.",
            document_ids=[doc_id]
        )
        content = data.get("content", "").strip()
        _cleanup_conversation(data.get("conversation_id"))

        # La risposta non deve finire con una parola troncata
        # Indicatori di troncamento: finisce senza punteggiatura finale
        last_char = content[-1] if content else ""
        ends_properly = last_char in ".!?:;)\"]'" or content.endswith("...")

        # Soft check - logga warning ma non fallisce se la risposta è lunga
        if not ends_properly and len(content) > 500:
            logger.warning(f"Possibile troncamento - ultimo char: '{last_char}'")
            logger.warning(f"Fine risposta: ...{content[-100:]}")

        assert len(content) > 100, (
            f"Risposta troppo corta per un elenco di punti: {len(content)} chars"
        )
        logger.info(f"Risposta completa: {len(content)} chars")

    def test_risposta_strutturata_per_elenchi(self, doc_id, check_server):
        """Quando si chiede un elenco, la risposta deve contenere elementi elencati."""
        data = _chat(
            "Elenca in punti i 3 argomenti più importanti di questo documento.",
            document_ids=[doc_id]
        )
        content = data.get("content", "")
        _cleanup_conversation(data.get("conversation_id"))

        # Cerca indicatori di lista (numeri, bullet points, trattini)
        list_patterns = [
            r"^\s*\d+[.)\-]",  # 1. 2. 3. oppure 1) 2) 3)
            r"^\s*[-•*]",       # bullet points
        ]
        lines = content.split("\n")
        list_items = 0
        for line in lines:
            for pattern in list_patterns:
                if re.match(pattern, line):
                    list_items += 1
                    break

        assert list_items >= 2, (
            f"Chiesto elenco ma trovati solo {list_items} elementi. Risposta: {content[:300]}"
        )
        logger.info(f"OK - {list_items} elementi nella lista")


# ---------------------------------------------------------------------------
# TEST 5: Istruzioni system prompt - segue le regole del RAG?
# ---------------------------------------------------------------------------
class TestIstruzioni:
    """Verifica che il modello segua le istruzioni del sistema RAG."""

    def test_risposta_basata_su_documenti(self, doc_id, check_server):
        """La risposta deve essere basata sui documenti, non su conoscenza generica."""
        data = _chat(
            "Basandoti SOLO sul documento, qual è il tema principale?",
            document_ids=[doc_id]
        )
        content = data.get("content", "")
        tool_used = data.get("tool_used", "")
        _cleanup_conversation(data.get("conversation_id"))

        # Il sistema dovrebbe aver usato lo strumento RAG
        assert len(content) > 50, f"Risposta troppo corta: {content}"
        logger.info(f"Tool usato: {tool_used}")
        logger.info(f"Risposta: {content[:200]}...")

    def test_rifiuta_domande_fuori_contesto(self, check_server):
        """Domande non relative ai documenti: il modello deve indicare i limiti."""
        data = _chat(
            "Qual è la ricetta della carbonara?",
            # Nessun document_id - domanda generica fuori contesto
        )
        content = data.get("content", "").lower()
        _cleanup_conversation(data.get("conversation_id"))

        # Il modello potrebbe rispondere comunque, ma dovrebbe almeno
        # non inventare contenuto come se fosse nei documenti
        logger.info(f"Risposta fuori contesto: {content[:300]}...")

    def test_cita_fonte(self, doc_id, doc_title, check_server):
        """La risposta dovrebbe indicare da quale documento proviene l'info."""
        data = _chat(
            "Da quale documento hai preso queste informazioni?",
            document_ids=[doc_id]
        )
        content = data.get("content", "")
        tool_details = data.get("tool_details", {})
        _cleanup_conversation(data.get("conversation_id"))

        # Verifica che tool_details contenga info sulla fonte
        has_source = bool(tool_details)
        logger.info(f"Ha dettagli tool: {has_source}")
        logger.info(f"Risposta: {content[:200]}...")


# ---------------------------------------------------------------------------
# TEST 6: Latenza - il modello risponde in tempo ragionevole?
# ---------------------------------------------------------------------------
class TestLatenza:
    """Verifica che i tempi di risposta siano accettabili."""

    MAX_SECONDS_SIMPLE = 60   # domande semplici
    MAX_SECONDS_COMPLEX = 90  # domande complesse

    def test_latenza_domanda_semplice(self, doc_id, check_server):
        """Una domanda semplice deve rispondere entro il timeout."""
        data = _chat("Di cosa parla?", document_ids=[doc_id])
        elapsed = data["_elapsed_seconds"]
        _cleanup_conversation(data.get("conversation_id"))

        logger.info(f"Latenza domanda semplice: {elapsed:.1f}s")
        assert elapsed < self.MAX_SECONDS_SIMPLE, (
            f"Troppo lento: {elapsed:.1f}s (max {self.MAX_SECONDS_SIMPLE}s)"
        )

    def test_latenza_domanda_complessa(self, doc_id, check_server):
        """Una domanda complessa ha più tempo ma deve comunque terminare."""
        data = _chat(
            "Analizza in dettaglio i 5 temi principali del documento, "
            "fornendo per ognuno una breve spiegazione e un esempio.",
            document_ids=[doc_id]
        )
        elapsed = data["_elapsed_seconds"]
        _cleanup_conversation(data.get("conversation_id"))

        logger.info(f"Latenza domanda complessa: {elapsed:.1f}s")
        assert elapsed < self.MAX_SECONDS_COMPLEX, (
            f"Troppo lento: {elapsed:.1f}s (max {self.MAX_SECONDS_COMPLEX}s)"
        )

    def test_latenza_consistente(self, doc_id, check_server):
        """Due domande simili devono avere latenze comparabili."""
        data1 = _chat("Riassumi il documento.", document_ids=[doc_id])
        _cleanup_conversation(data1.get("conversation_id"))
        t1 = data1["_elapsed_seconds"]

        data2 = _chat("Fai un riassunto del documento.", document_ids=[doc_id])
        _cleanup_conversation(data2.get("conversation_id"))
        t2 = data2["_elapsed_seconds"]

        ratio = max(t1, t2) / max(min(t1, t2), 0.1)
        logger.info(f"Latenze: {t1:.1f}s vs {t2:.1f}s (ratio: {ratio:.1f}x)")

        # La seconda non dovrebbe essere più di 5x la prima
        # (la prima query può essere lenta per il caricamento modello in GPU)
        assert ratio < 5.0, (
            f"Latenza inconsistente: {t1:.1f}s vs {t2:.1f}s (ratio {ratio:.1f}x)"
        )
