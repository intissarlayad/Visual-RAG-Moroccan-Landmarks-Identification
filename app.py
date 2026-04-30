import streamlit as st
import json
import numpy as np
from pathlib import Path
from PIL import Image
import requests
import chromadb
from sentence_transformers import SentenceTransformer
import datetime
import time
import threading
import pandas as pd
import urllib.parse

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "CHROMA_PATH"     : Path("chroma_db"),
    "COLLECTION_NAME" : "lieux_touristiques_maroc",
    "CLIP_MODEL"      : "clip-ViT-B-32",
    "OLLAMA_URL"      : "http://localhost:11434",
    "METADATA_PATH"   : Path("metadata.json"),
    "DATASET_PATH"    : Path("dataset"),
}

MODELES_DISPONIBLES = ["qwen2.5:1.5b", "mistral", "llama3", "llama3.2"]

LANGUES = {
    "Francais" : "français",
    "English"  : "english",
    "Darija"   : "darija marocaine (arabe dialectal marocain)",
}

SYSTEM_PROMPTS = {
    "français": "Tu es un guide touristique expert du Maroc. Reponds en francais. Structure : presentation -> histoire -> anecdote -> conseil visite. Utilise UNIQUEMENT les informations fournies.",
    "english" : "You are an expert Moroccan tourist guide. Always respond in English. Structure: presentation -> history -> anecdote -> visit tip. Use ONLY the provided information.",
    "darija marocaine (arabe dialectal marocain)": "nta mrchid siyahi khabir dyal lmaghrib. jawb dima bdarija. istikhdam ghir lmaalumat lli atawak.",
}

SEUIL_DEFAULT = 0.40

# ── CSS ────────────────────────────────────────────────────────────────────────
def inject_css(dark_mode: bool):
    if dark_mode:
        bg = "#0f0a05"; card_bg = "#1a1108"; text = "#e8d5b0"; subtext = "#a08060"; border = "#3a2510"
        header_bg = "linear-gradient(135deg,#0f0a05 0%,#2a1200 60%,#4a2000 100%)"
        fiche_bg = "#1a1108"
    else:
        bg = "#fdf8f2"; card_bg = "#fff8ee"; text = "#2d1f0e"; subtext = "#7a5c3a"; border = "#e8d5b0"
        header_bg = "linear-gradient(135deg,#1a0a00 0%,#4a1a00 50%,#8b3a0f 100%)"
        fiche_bg = "#fffbf5"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');
    html,body,[class*="css"]{{font-family:'DM Sans',sans-serif;background-color:{bg};color:{text};}}
    h1,h2,h3{{font-family:'Playfair Display',serif !important;}}
    .main-header{{background:{header_bg};padding:2rem 2.5rem;border-radius:16px;margin-bottom:1.5rem;color:white;position:relative;overflow:hidden;}}
    .main-header::before{{content:'☽✦';position:absolute;right:2rem;top:50%;transform:translateY(-50%);font-size:4rem;opacity:.12;letter-spacing:-.5rem;}}
    .main-header h1{{color:#f5c87a !important;font-size:2rem;margin:0;}}
    .main-header p{{color:#e8d5b0;margin:.4rem 0 0;font-size:.95rem;}}
    .landing-card{{background:{card_bg};border:1px solid {border};border-radius:14px;padding:1rem;text-align:center;transition:transform .2s,box-shadow .2s;}}
    .landing-card:hover{{transform:translateY(-4px);box-shadow:0 8px 25px rgba(139,58,15,.2);}}
    .landing-card h4{{font-family:'Playfair Display',serif;color:#c17a2a;margin:.5rem 0 .2rem;font-size:1rem;}}
    .landing-card p{{font-size:.78rem;color:{subtext};margin:0;}}
    .badge-confiant{{background:#d4edda;color:#155724;padding:3px 12px;border-radius:20px;font-size:.82rem;font-weight:600;}}
    .badge-probable{{background:#fff3cd;color:#856404;padding:3px 12px;border-radius:20px;font-size:.82rem;font-weight:600;}}
    .badge-incertain{{background:#f8d7da;color:#721c24;padding:3px 12px;border-radius:20px;font-size:.82rem;font-weight:600;}}
    .badge-inconnu{{background:#e2e3e5;color:#383d41;padding:3px 12px;border-radius:20px;font-size:.82rem;font-weight:600;}}
    .badge-web{{background:#cce5ff;color:#004085;padding:3px 12px;border-radius:20px;font-size:.82rem;font-weight:600;}}
    .fiche-box{{background:{fiche_bg};border-left:4px solid #c17a2a;border-radius:0 12px 12px 0;padding:1.2rem 1.5rem;margin-top:.8rem;line-height:1.75;color:{text};}}
    .fiche-web{{background:{fiche_bg};border-left:4px solid #0d6efd;border-radius:0 12px 12px 0;padding:1.2rem 1.5rem;margin-top:.8rem;line-height:1.75;color:{text};}}
    .hist-item{{background:{card_bg};border:1px solid {border};border-radius:10px;padding:.6rem 1rem;margin-bottom:.4rem;font-size:.85rem;}}
    .stat-card{{background:{card_bg};border:1px solid {border};border-radius:12px;padding:1rem 1.2rem;text-align:center;}}
    .stat-card h2{{color:#c17a2a;font-family:'Playfair Display',serif;margin:0;font-size:2rem;}}
    .stat-card p{{color:{subtext};margin:.2rem 0 0;font-size:.82rem;}}
    .web-info-box{{background:{card_bg};border:2px dashed #0d6efd;border-radius:12px;padding:1rem 1.2rem;margin:.5rem 0;}}
    .stButton>button{{background:linear-gradient(135deg,#8b3a0f,#c17a2a) !important;color:white !important;border:none !important;border-radius:10px !important;font-weight:500 !important;transition:transform .15s !important;}}
    .stButton>button:hover{{transform:translateY(-2px) !important;box-shadow:0 4px 15px rgba(139,58,15,.35) !important;}}
    .stProgress>div>div{{background:linear-gradient(90deg,#8b3a0f,#f5c87a) !important;border-radius:8px !important;}}
    </style>
    """, unsafe_allow_html=True)


# ── Ressources ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_clip():
    return SentenceTransformer(CONFIG["CLIP_MODEL"])

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=str(CONFIG["CHROMA_PATH"]))
    return client.get_or_create_collection(
        name=CONFIG["COLLECTION_NAME"],
        metadata={"hnsw:space": "cosine"},
    )

def load_metadata() -> dict:
    if CONFIG["METADATA_PATH"].exists():
        with open(CONFIG["METADATA_PATH"], encoding="utf-8") as f:
            return json.load(f)
    return {"lieux": []}


# ── Reindexation ───────────────────────────────────────────────────────────────
def reindexer(clip_model, collection, metadata: dict) -> int:
    count = 0
    for lieu in metadata.get("lieux", []):
        for img_rel in lieu.get("images", []):
            img_path = CONFIG["DATASET_PATH"] / img_rel
            if not img_path.exists():
                continue
            try:
                image = Image.open(img_path).convert("RGB")
                embedding = clip_model.encode(image).tolist()
                doc_id = f"{lieu['id']}_{img_path.stem}"
                collection.upsert(
                    ids=[doc_id],
                    embeddings=[embedding],
                    metadatas=[{
                        "nom"      : lieu.get("nom",""),
                        "ville"    : lieu.get("ville",""),
                        "histoire" : lieu.get("histoire",""),
                        "epoque"   : lieu.get("epoque",""),
                        "style"    : lieu.get("style",""),
                        "image_ref": str(img_path),
                        "lat"      : lieu.get("coordonnees",{}).get("lat",0.0),
                        "lon"      : lieu.get("coordonnees",{}).get("lon",0.0),
                    }],
                    documents=[lieu.get("nom","")],
                )
                count += 1
            except Exception as e:
                st.warning(f"Erreur {img_path.name}: {e}")
    return count


# ── Recherche locale ───────────────────────────────────────────────────────────
def rechercher_lieu(image: Image.Image, clip_model, collection, n_results: int = 3):
    embedding = clip_model.encode(image).tolist()
    raw = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["metadatas","documents","distances"],
    )
    results = []
    for rang,(meta,dist) in enumerate(zip(raw["metadatas"][0],raw["distances"][0]),start=1):
        results.append({
            "rang"     : rang,
            "nom"      : meta.get("nom","Inconnu"),
            "ville"    : meta.get("ville","N/A"),
            "histoire" : meta.get("histoire",""),
            "epoque"   : meta.get("epoque",""),
            "style"    : meta.get("style",""),
            "image_ref": meta.get("image_ref",""),
            "lat"      : meta.get("lat",0.0),
            "lon"      : meta.get("lon",0.0),
            "score"    : round(1 - dist, 4),
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# RECHERCHE WEB — Wikipedia API + description via LLM
# ══════════════════════════════════════════════════════════════════════════════

def identifier_lieu_via_llm(image: Image.Image, modele: str, timeout: int) -> str:
    """
    Envoie l'image au LLM via Ollama (modele vision si disponible)
    pour identifier le nom du lieu. Fallback : prompt texte simple.
    """
    # On utilise LLaVA si disponible, sinon on demande à l'utilisateur de nommer le lieu
    # Ici on fait un appel simple pour identifier le lieu depuis le nom candidat
    payload = {
        "model"   : modele,
        "messages": [{
            "role"   : "user",
            "content": (
                "Tu es un expert en monuments et lieux touristiques du monde entier, "
                "specialise au Maroc. Je vais te decrire une situation : "
                "une photo a ete soumise a un systeme de recherche d'images, "
                "mais le lieu n'a pas ete reconnu dans la base de donnees locale. "
                "Dis-moi quel type de monument ou lieu je devrais rechercher sur Wikipedia "
                "pour un lieu touristique marocain generique. "
                "Reponds avec UNIQUEMENT le nom du lieu le plus probable en 2-4 mots, rien d'autre."
            ),
        }],
        "stream"  : False,
        "options" : {"temperature": 0.3, "num_predict": 20},
    }
    try:
        resp = requests.post(f"{CONFIG['OLLAMA_URL']}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except:
        return ""


def rechercher_wikipedia(nom_lieu: str, langue: str = "fr") -> dict:
    """
    Cherche un lieu sur Wikipedia et retourne titre + extrait + URL + coordonnées.
    """
    lang_wiki = "fr" if "franc" in langue else "en"

    # 1. Recherche du titre exact
    search_url = f"https://{lang_wiki}.wikipedia.org/w/api.php"
    params_search = {
        "action"  : "query",
        "list"    : "search",
        "srsearch": nom_lieu + " Maroc monument",
        "format"  : "json",
        "srlimit" : 3,
    }
    try:
        resp = requests.get(search_url, params=params_search, timeout=10)
        data = resp.json()
        results = data.get("query", {}).get("search", [])
        if not results:
            return {}
        titre = results[0]["title"]
    except:
        return {}

    # 2. Récupération du résumé + coordonnées
    params_extract = {
        "action"     : "query",
        "titles"     : titre,
        "prop"       : "extracts|coordinates|pageimages",
        "exintro"    : True,
        "explaintext": True,
        "exsentences": 6,
        "coprop"     : "type",
        "format"     : "json",
        "pithumbsize": 400,
    }
    try:
        resp2 = requests.get(search_url, params=params_extract, timeout=10)
        data2 = resp2.json()
        pages = data2.get("query", {}).get("pages", {})
        page  = next(iter(pages.values()))

        extrait = page.get("extract", "")
        coords  = page.get("coordinates", [{}])[0]
        thumb   = page.get("thumbnail", {}).get("source", "")
        url_enc = urllib.parse.quote(titre.replace(" ", "_"))
        url     = f"https://{lang_wiki}.wikipedia.org/wiki/{url_enc}"

        return {
            "titre"  : titre,
            "extrait": extrait[:1500] if extrait else "Aucune description disponible.",
            "lat"    : coords.get("lat", 0.0),
            "lon"    : coords.get("lon", 0.0),
            "image"  : thumb,
            "url"    : url,
            "source" : "Wikipedia",
        }
    except:
        return {}


def rechercher_web_complet(nom_saisi: str, langue: str) -> dict:
    """
    Pipeline de recherche web : Wikipedia FR + EN en fallback.
    """
    lang_wiki = "fr" if "franc" in langue else "en"

    # Tentative dans la langue choisie
    resultat = rechercher_wikipedia(nom_saisi, lang_wiki)
    if resultat:
        return resultat

    # Fallback en anglais
    if lang_wiki != "en":
        resultat = rechercher_wikipedia(nom_saisi, "en")
        if resultat:
            return resultat

    return {}


def generer_fiche_depuis_web(contexte_web: dict, question: str, modele: str, langue: str, timeout: int) -> str:
    """
    Génère une fiche touristique à partir des données Wikipedia via Ollama.
    """
    system = SYSTEM_PROMPTS.get(langue, SYSTEM_PROMPTS["français"])
    prompt = (
        "=== INFORMATIONS WEB (Wikipedia) ===\n"
        f"Titre   : {contexte_web.get('titre','')}\n"
        f"Source  : {contexte_web.get('source','Wikipedia')}\n"
        f"Resume  :\n{contexte_web.get('extrait','')}\n"
        "=====================================\n\n"
        f"QUESTION : {question}\n\n"
        "Note : Ce lieu n'etait pas dans la base locale. "
        "Genere une fiche touristique elegante a partir du resume ci-dessus."
    )
    payload = {
        "model"   : modele,
        "messages": [
            {"role":"system","content":system},
            {"role":"user","content":prompt},
        ],
        "stream"  : False,
        "options" : {"temperature":0.7,"num_predict":400},
    }
    try:
        resp = requests.post(f"{CONFIG['OLLAMA_URL']}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.Timeout:
        return f"Timeout ({timeout}s). Essayez qwen2.5:1.5b."
    except Exception as e:
        return f"Erreur generation : {e}"


# ── Badge ──────────────────────────────────────────────────────────────────────
def badge_confiance(score: float, seuil: float) -> str:
    if score < seuil:
        return '<span class="badge-inconnu">Lieu non reconnu localement</span>'
    elif score >= 0.80:
        return '<span class="badge-confiant">Tres confiant</span>'
    elif score >= 0.55:
        return '<span class="badge-probable">Probable</span>'
    else:
        return '<span class="badge-incertain">Incertain</span>'


# ── Generation LLM locale ──────────────────────────────────────────────────────
def generate_response(context: dict, question: str, modele: str, langue: str, timeout: int) -> str:
    system = SYSTEM_PROMPTS.get(langue, SYSTEM_PROMPTS["français"])
    prompt = (
        "=== LIEU IDENTIFIE ===\n"
        f"Nom     : {context['nom']}\n"
        f"Ville   : {context['ville']}\n"
        f"Epoque  : {context.get('epoque','N/A')}\n"
        f"Style   : {context.get('style','N/A')}\n"
        f"Histoire:\n{context['histoire']}\n"
        "====================\n\n"
        f"QUESTION : {question}"
    )
    payload = {
        "model"   : modele,
        "messages": [{"role":"system","content":system},{"role":"user","content":prompt}],
        "stream"  : False,
        "options" : {"temperature":0.7,"num_predict":400},
    }
    try:
        resp = requests.post(f"{CONFIG['OLLAMA_URL']}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.Timeout:
        return f"Timeout ({timeout}s). Essayez qwen2.5:1.5b."
    except Exception as e:
        return f"Erreur : {e}"


# ── Generation avec thread + barre progression ─────────────────────────────────
def generer_avec_progression(fn_generate, timeout: int):
    prog_bar = st.progress(0, text="En attente du LLM...")
    t_debut  = time.time()
    reponse  = [None]

    def worker():
        reponse[0] = fn_generate()

    t = threading.Thread(target=worker)
    t.start()
    while t.is_alive():
        elapsed = time.time() - t_debut
        pct = min(int((elapsed / timeout) * 100), 95)
        prog_bar.progress(pct, text=f"Generation en cours... {elapsed:.0f}s")
        time.sleep(0.5)
    t.join()
    elapsed_total = time.time() - t_debut
    prog_bar.progress(100, text=f"Genere en {elapsed_total:.1f}s")
    time.sleep(0.3)
    prog_bar.empty()
    return reponse[0] or "Erreur lors de la generation."


# ── Export ─────────────────────────────────────────────────────────────────────
def export_txt(nom: str, ville: str, score_str: str, reponse: str, source: str = "Base locale") -> bytes:
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    return (
        f"FICHE TOURISTIQUE - Visual RAG Maroc\n"
        f"Genere le : {now}\n"
        f"Source    : {source}\n"
        f"{'='*50}\n\n"
        f"Lieu  : {nom}\nVille : {ville}\nScore : {score_str}\n\n"
        f"{'='*50}\n\n{reponse}\n"
    ).encode("utf-8")


# ── Page accueil ───────────────────────────────────────────────────────────────
def page_accueil(metadata: dict):
    st.markdown("### Lieux disponibles dans la base locale")
    st.caption("Ces lieux sont indexes et identifiables directement. Tout autre lieu sera recherche en ligne.")
    lieux = metadata.get("lieux", [])
    if not lieux:
        st.info("Aucun lieu dans metadata.json.")
        return
    cols = st.columns(min(len(lieux), 3))
    for i, lieu in enumerate(lieux):
        with cols[i % 3]:
            for img_rel in lieu.get("images", []):
                img_path = CONFIG["DATASET_PATH"] / img_rel
                if img_path.exists():
                    try:
                        st.image(Image.open(img_path), use_container_width=True)
                        break
                    except:
                        pass
            st.markdown(f"""
            <div class="landing-card">
                <h4>{lieu.get('nom','')}</h4>
                <p>{lieu.get('ville','')} | {lieu.get('epoque','')}</p>
                <p>{lieu.get('style','')} | {lieu.get('classification','')}</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")


# ── Page stats ─────────────────────────────────────────────────────────────────
def page_statistiques(historique: list, metadata: dict, clip_model, collection):
    st.markdown("### Statistiques de la session")
    if not historique:
        st.info("Aucune recherche effectuee dans cette session.")
    else:
        scores = [h["score"] for h in historique if h["score"] > 0]
        noms   = [h["nom"] for h in historique]
        top    = max(set(noms), key=noms.count) if noms else "N/A"
        c1, c2, c3, c4 = st.columns(4)
        nb_web   = sum(1 for h in historique if h.get("source") == "Wikipedia")
        nb_local = len(historique) - nb_web
        with c1:
            st.markdown(f'<div class="stat-card"><h2>{len(historique)}</h2><p>Recherches totales</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><h2>{nb_local}</h2><p>Identifiees en local</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-card"><h2>{nb_web}</h2><p>Recherchees en ligne</p></div>', unsafe_allow_html=True)
        with c4:
            moy = f"{np.mean(scores):.0%}" if scores else "N/A"
            st.markdown(f'<div class="stat-card"><h2>{moy}</h2><p>Score moyen local</p></div>', unsafe_allow_html=True)

        st.markdown("")
        df = pd.DataFrame(historique)
        st.markdown("**Lieux identifies**")
        st.bar_chart(df["nom"].value_counts())
        if scores:
            st.markdown("**Scores de confiance**")
            st.line_chart(pd.DataFrame({"Score": scores}))

    st.divider()
    st.markdown("### Test de performance sur le dataset")
    if st.button("Lancer le test de performance", use_container_width=True):
        lieux = metadata.get("lieux", [])
        resultats_test = []
        total = sum(len(l.get("images",[])) for l in lieux)
        barre = st.progress(0)
        done  = 0
        for lieu in lieux:
            for img_rel in lieu.get("images", []):
                img_path = CONFIG["DATASET_PATH"] / img_rel
                if not img_path.exists():
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                    res = rechercher_lieu(img, clip_model, collection, n_results=1)
                    if res:
                        correct = lieu["nom"].lower() in res[0]["nom"].lower()
                        resultats_test.append({
                            "Attendu": lieu["nom"],
                            "Predit" : res[0]["nom"],
                            "Score"  : f"{res[0]['score']:.2%}",
                            "Correct": "OK" if correct else "ERREUR",
                        })
                except:
                    pass
                done += 1
                barre.progress(done / max(total,1))
        barre.empty()
        if resultats_test:
            df_res = pd.DataFrame(resultats_test)
            precision = (df_res["Correct"] == "OK").mean()
            st.success(f"Precision@1 : {precision:.2%} sur {len(df_res)} images")
            st.dataframe(df_res, use_container_width=True, hide_index=True)


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Visual RAG - Tourisme Marocain",
        page_icon="🕌",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "historique" not in st.session_state:
        st.session_state.historique = []
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

    inject_css(st.session_state.dark_mode)

    st.markdown("""
    <div class="main-header">
        <h1>🕌 Visual RAG — Tourisme Marocain</h1>
        <p>Identifiez un lieu par photo · Base locale + Recherche Wikipedia automatique si lieu inconnu</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Configuration")

        dark = st.toggle("Mode sombre", value=st.session_state.dark_mode)
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()

        st.divider()
        modele_choisi  = st.selectbox("Modele LLM", MODELES_DISPONIBLES, help="qwen2.5:1.5b = plus rapide")
        langue_label   = st.selectbox("Langue de reponse", list(LANGUES.keys()))
        langue_choisie = LANGUES[langue_label]
        n_results      = st.slider("Lieux candidats", 1, 5, 3)
        seuil          = st.slider("Seuil de confiance (local)", 0.0, 1.0, SEUIL_DEFAULT, 0.05,
                                    help="En dessous = recherche automatique sur Wikipedia")
        timeout_llm    = st.slider("Timeout LLM (s)", 60, 600, 240, 30)

        st.divider()
        with st.spinner("Chargement CLIP..."):
            clip_model = load_clip()
        with st.spinner("ChromaDB..."):
            collection = load_collection()
        metadata = load_metadata()
        n_docs   = collection.count()

        if n_docs == 0:
            st.error("Base vide - reindexez !")
        else:
            st.success(f"{n_docs} images indexees en local")

        st.info("Si le lieu n'est pas dans la base, Wikipedia est consulte automatiquement.")

        st.divider()
        if st.button("Reindexer le dataset", use_container_width=True):
            with st.spinner("Indexation..."):
                nb = reindexer(clip_model, collection, metadata)
            st.success(f"{nb} images reindexees !")
            st.rerun()

        st.divider()
        st.caption(f"CLIP: clip-ViT-B-32 | LLM: {modele_choisi}")
        st.caption("Visual RAG Maroc — Local + Web")

    # ── Onglets ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Identification", "Lieux disponibles", "Statistiques"])

    # ── TAB 1 ──────────────────────────────────────────────────────────────────
    with tab1:
        col_input, col_output = st.columns([1, 1], gap="large")

        with col_input:
            st.subheader("Image a identifier")
            mode = st.radio("Source", ["Uploader un fichier", "Prendre une photo"], horizontal=True)
            uploaded = None
            if mode == "Uploader un fichier":
                uploaded = st.file_uploader("Photo JPG/PNG", type=["jpg","jpeg","png"],
                                             label_visibility="collapsed")
            else:
                uploaded = st.camera_input("Prenez une photo")

            question = st.text_area(
                "Question",
                value="Parle-moi de ce lieu. Quelle est son histoire et que faut-il voir ?",
                height=90,
            )

            # Champ optionnel pour aider la recherche web
            nom_manuel = st.text_input(
                "Nom du lieu (optionnel, aide la recherche Wikipedia)",
                placeholder="Ex: Mosquee Hassan II, Chefchaouen, Merzouga...",
                help="Si votre lieu n'est pas reconnu localement, ce nom sera utilise pour chercher sur Wikipedia."
            )

            go = st.button("Identifier et generer la fiche", type="primary", use_container_width=True)
            if uploaded:
                st.image(uploaded, caption="Image selectionnee", use_container_width=True)

        with col_output:
            st.subheader("Resultats")

            if go and uploaded:
                image = Image.open(uploaded).convert("RGB")

                # ── ETAPE 1 : Recherche locale ─────────────────────────────
                with st.spinner("Recherche dans la base locale..."):
                    resultats = rechercher_lieu(image, clip_model, collection, n_results) if n_docs > 0 else []

                lieu_local_trouve = resultats and resultats[0]["score"] >= seuil

                if resultats:
                    st.markdown("**Lieux candidats (base locale) :**")
                    for r in resultats:
                        pct = max(0, min(100, int(r["score"]*100)))
                        st.progress(pct, text=f"#{r['rang']} {r['nom']} ({r['ville']}) - {r['score']:.4f}")
                    st.divider()

                # ══════════════════════════════════════════════════════════
                # CAS 1 : LIEU TROUVE EN LOCAL
                # ══════════════════════════════════════════════════════════
                if lieu_local_trouve:
                    lieu = resultats[0]

                    st.markdown('<span class="badge-confiant">Base locale</span>', unsafe_allow_html=True)

                    # Comparaison cote a cote
                    st.markdown("**Votre photo vs reference du dataset**")
                    c_up, c_ref = st.columns(2)
                    with c_up:
                        st.image(image, caption="Votre photo", use_container_width=True)
                    with c_ref:
                        ref_path = lieu.get("image_ref","")
                        if ref_path and Path(ref_path).exists():
                            try:
                                st.image(Image.open(ref_path), caption=f"Reference : {lieu['nom']}", use_container_width=True)
                            except:
                                st.caption("Reference non disponible")
                        else:
                            st.caption("Reference non disponible")

                    st.divider()
                    c_nom, c_badge = st.columns([3,2])
                    with c_nom:
                        st.markdown(f"**{lieu['nom']} — {lieu['ville']}**")
                        if lieu.get("epoque"):
                            st.caption(f"{lieu['epoque']}  |  {lieu.get('style','')}")
                    with c_badge:
                        st.markdown(badge_confiance(lieu["score"], seuil), unsafe_allow_html=True)
                        st.caption(f"Score : {lieu['score']:.2%}")

                    if lieu.get("lat") and lieu["lat"] != 0:
                        st.map(pd.DataFrame([{"lat": lieu["lat"], "lon": lieu["lon"]}]), zoom=8)

                    # Generation fiche locale
                    reponse_finale = generer_avec_progression(
                        lambda: generate_response(lieu, question, modele_choisi, langue_choisie, timeout_llm),
                        timeout_llm
                    )
                    st.markdown("**Fiche du guide touristique IA :**")
                    st.markdown(f'<div class="fiche-box">{reponse_finale}</div>', unsafe_allow_html=True)

                    # Actions
                    st.divider()
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.download_button("Telecharger la fiche",
                            data=export_txt(lieu["nom"], lieu["ville"], f"{lieu['score']:.2%}", reponse_finale, "Base locale"),
                            file_name=f"fiche_{lieu['nom'].replace(' ','_')}.txt",
                            mime="text/plain", use_container_width=True)
                    with c2:
                        if st.button("Correct", use_container_width=True):
                            st.toast("Merci !")
                    with c3:
                        if st.button("Incorrect", use_container_width=True):
                            st.toast("Retour note.")

                    st.text_area("Copier la fiche", value=reponse_finale, height=100)

                    # Historique
                    st.session_state.historique.insert(0, {
                        "nom"   : lieu["nom"],
                        "ville" : lieu["ville"],
                        "score" : lieu["score"],
                        "source": "Base locale",
                        "heure" : datetime.datetime.now().strftime("%H:%M:%S"),
                    })

                # ══════════════════════════════════════════════════════════
                # CAS 2 : LIEU NON RECONNU → RECHERCHE WEB AUTOMATIQUE
                # ══════════════════════════════════════════════════════════
                else:
                    st.warning(f"Lieu non reconnu dans la base locale (score max : {resultats[0]['score']:.2%} < seuil {seuil:.0%})" if resultats else "Base locale vide ou lieu inconnu.")
                    st.markdown("---")
                    st.markdown("### Recherche automatique en ligne...")

                    # Nom pour la recherche
                    nom_recherche = nom_manuel.strip() if nom_manuel.strip() else "monument touristique Maroc"

                    st.info(f"Recherche Wikipedia pour : **{nom_recherche}**")

                    with st.spinner("Consultation de Wikipedia..."):
                        contexte_web = rechercher_web_complet(nom_recherche, langue_choisie)

                    if contexte_web:
                        st.success(f"Trouve sur Wikipedia : **{contexte_web['titre']}**")

                        # Infos Wikipedia
                        st.markdown(f"""
                        <div class="web-info-box">
                            <b>Source :</b> Wikipedia &nbsp;|&nbsp;
                            <b>Titre :</b> {contexte_web['titre']} &nbsp;|&nbsp;
                            <a href="{contexte_web['url']}" target="_blank">Voir sur Wikipedia</a>
                        </div>
                        """, unsafe_allow_html=True)

                        # Image Wikipedia si disponible
                        if contexte_web.get("image"):
                            c_up2, c_wiki = st.columns(2)
                            with c_up2:
                                st.image(image, caption="Votre photo", use_container_width=True)
                            with c_wiki:
                                try:
                                    st.image(contexte_web["image"], caption=f"Wikipedia : {contexte_web['titre']}", use_container_width=True)
                                except:
                                    pass

                        # Badge + score
                        c_nom2, c_badge2 = st.columns([3,2])
                        with c_nom2:
                            st.markdown(f"**{contexte_web['titre']}**")
                        with c_badge2:
                            st.markdown('<span class="badge-web">Recherche Web</span>', unsafe_allow_html=True)

                        # Carte GPS Wikipedia
                        if contexte_web.get("lat") and contexte_web["lat"] != 0:
                            st.map(pd.DataFrame([{"lat": contexte_web["lat"], "lon": contexte_web["lon"]}]), zoom=8)

                        # Résumé Wikipedia brut
                        with st.expander("Voir le resume Wikipedia brut"):
                            st.write(contexte_web["extrait"])

                        # Generation fiche depuis les données Wikipedia
                        st.markdown("**Generation de la fiche a partir de Wikipedia...**")
                        reponse_web = generer_avec_progression(
                            lambda ctx=contexte_web: generer_fiche_depuis_web(ctx, question, modele_choisi, langue_choisie, timeout_llm),
                            timeout_llm
                        )
                        st.markdown("**Fiche du guide touristique IA (source Wikipedia) :**")
                        st.markdown(f'<div class="fiche-web">{reponse_web}</div>', unsafe_allow_html=True)

                        # Actions
                        st.divider()
                        c1w, c2w = st.columns(2)
                        with c1w:
                            st.download_button("Telecharger la fiche",
                                data=export_txt(contexte_web["titre"], "", "Wikipedia", reponse_web, "Wikipedia"),
                                file_name=f"fiche_{contexte_web['titre'].replace(' ','_')}.txt",
                                mime="text/plain", use_container_width=True)
                        with c2w:
                            st.link_button("Voir sur Wikipedia", contexte_web["url"], use_container_width=True)

                        st.text_area("Copier la fiche", value=reponse_web, height=100)

                        # Historique
                        st.session_state.historique.insert(0, {
                            "nom"   : contexte_web["titre"],
                            "ville" : "Wikipedia",
                            "score" : 0.0,
                            "source": "Wikipedia",
                            "heure" : datetime.datetime.now().strftime("%H:%M:%S"),
                        })

                    else:
                        st.error("Aucun resultat trouve sur Wikipedia.")
                        st.markdown(f"""
                        **Suggestions :**
                        - Precisez le nom du lieu dans le champ **"Nom du lieu"** a gauche
                        - Verifiez l'orthographe (ex: "Mosquee Hassan II", "Ait Benhaddou", "Chefchaouen")
                        - Essayez en anglais (ex: "Hassan II Mosque", "Blue City Morocco")
                        """)

                    st.markdown("""
                    ---
                    **Ce lieu n'est pas dans la base locale.** Pour l'ajouter :
                    1. Ajoutez des photos dans `dataset/<nom_du_lieu>/`
                    2. Completez `metadata.json` avec les informations du lieu
                    3. Cliquez **Reindexer le dataset** dans la sidebar
                    """)

                # Historique session (commun)
                st.session_state.historique = st.session_state.historique[:15]

            elif go and not uploaded:
                st.warning("Choisissez une image d'abord.")
            else:
                st.info("Uploadez une image et cliquez sur Identifier.")
                if st.session_state.historique:
                    st.markdown("**Dernieres recherches**")
                    for h in st.session_state.historique[:5]:
                        source_badge = "Wikipedia" if h.get("source") == "Wikipedia" else "Local"
                        st.markdown(f"""
                        <div class="hist-item">
                            <strong>{h['nom']}</strong> — {h['ville']}
                            &nbsp;|&nbsp; {source_badge}
                            &nbsp;|&nbsp; {h['heure']}
                        </div>
                        """, unsafe_allow_html=True)

    # ── TAB 2 ──────────────────────────────────────────────────────────────────
    with tab2:
        page_accueil(metadata)

    # ── TAB 3 ──────────────────────────────────────────────────────────────────
    with tab3:
        page_statistiques(st.session_state.historique, metadata, clip_model, collection)


if __name__ == "__main__":
    main()