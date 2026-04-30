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
    "Francais"  : "français",
    "English"   : "english",
    "Darija"    : "darija marocaine (arabe dialectal marocain)",
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
    .fiche-box{{background:{fiche_bg};border-left:4px solid #c17a2a;border-radius:0 12px 12px 0;padding:1.2rem 1.5rem;margin-top:.8rem;line-height:1.75;color:{text};}}
    .hist-item{{background:{card_bg};border:1px solid {border};border-radius:10px;padding:.6rem 1rem;margin-bottom:.4rem;font-size:.85rem;}}
    .stat-card{{background:{card_bg};border:1px solid {border};border-radius:12px;padding:1rem 1.2rem;text-align:center;}}
    .stat-card h2{{color:#c17a2a;font-family:'Playfair Display',serif;margin:0;font-size:2rem;}}
    .stat-card p{{color:{subtext};margin:.2rem 0 0;font-size:.82rem;}}
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


# ── Recherche ──────────────────────────────────────────────────────────────────
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


# ── Badge ──────────────────────────────────────────────────────────────────────
def badge_confiance(score: float, seuil: float) -> str:
    if score < seuil:
        return '<span class="badge-inconnu">Lieu non reconnu</span>'
    elif score >= 0.80:
        return '<span class="badge-confiant">Tres confiant</span>'
    elif score >= 0.55:
        return '<span class="badge-probable">Probable</span>'
    else:
        return '<span class="badge-incertain">Incertain</span>'


# ── Generation LLM ─────────────────────────────────────────────────────────────
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
    except requests.exceptions.ConnectionError:
        return "Erreur : Ollama n'est pas accessible. Lancez 'ollama serve'."
    except requests.exceptions.Timeout:
        return f"Erreur : Timeout depasse ({timeout}s). Essayez qwen2.5:1.5b."
    except Exception as e:
        return f"Erreur inattendue : {e}"


# ── Export ─────────────────────────────────────────────────────────────────────
def export_txt(lieu: dict, reponse: str) -> bytes:
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    return (
        f"FICHE TOURISTIQUE - Visual RAG Maroc\n"
        f"Genere le : {now}\n{'='*50}\n\n"
        f"Lieu   : {lieu['nom']}\nVille  : {lieu['ville']}\n"
        f"Epoque : {lieu.get('epoque','N/A')}\nStyle  : {lieu.get('style','N/A')}\n"
        f"Score  : {lieu['score']:.2%}\n\n{'='*50}\n\n{reponse}\n"
    ).encode("utf-8")


# ── Page accueil ───────────────────────────────────────────────────────────────
def page_accueil(metadata: dict):
    st.markdown("### Lieux disponibles dans la base")
    st.caption("Ces lieux sont indexes et identifiables par le systeme Visual RAG.")
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
        scores = [h["score"] for h in historique]
        c1, c2, c3 = st.columns(3)
        noms = [h["nom"] for h in historique]
        top  = max(set(noms), key=noms.count)
        with c1:
            st.markdown(f'<div class="stat-card"><h2>{len(historique)}</h2><p>Recherches</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><h2>{np.mean(scores):.0%}</h2><p>Score moyen</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-card"><h2 style="font-size:1rem">{top}</h2><p>Lieu le plus identifie</p></div>', unsafe_allow_html=True)

        st.markdown("")
        df_hist = pd.DataFrame(historique)

        st.markdown("**Lieux les plus identifies**")
        count_lieux = df_hist["nom"].value_counts().reset_index()
        count_lieux.columns = ["Lieu","Nombre"]
        st.bar_chart(count_lieux.set_index("Lieu"))

        st.markdown("**Scores au fil des recherches**")
        st.line_chart(pd.DataFrame({"Score": scores}))

    st.divider()
    st.markdown("### Matrice de confusion — Test sur le dataset complet")
    st.caption("Teste chaque image du dataset et verifie si le lieu est correctement identifie.")

    if st.button("Lancer le test de performance", use_container_width=True):
        lieux = metadata.get("lieux", [])
        resultats_test = []
        total_imgs = sum(len(l.get("images",[])) for l in lieux)
        barre = st.progress(0, text="Test en cours...")
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
                        predit  = res[0]["nom"]
                        score   = res[0]["score"]
                        correct = lieu["nom"].lower() in predit.lower()
                        resultats_test.append({
                            "Attendu": lieu["nom"],
                            "Predit" : predit,
                            "Score"  : f"{score:.2%}",
                            "Correct": "OK" if correct else "ERREUR",
                        })
                except:
                    pass
                done += 1
                barre.progress(done / max(total_imgs,1), text=f"Test {done}/{total_imgs}")
        barre.empty()
        if resultats_test:
            df_res    = pd.DataFrame(resultats_test)
            precision = (df_res["Correct"] == "OK").mean()
            st.success(f"Precision@1 : {precision:.2%} sur {len(df_res)} images")
            st.dataframe(df_res, use_container_width=True, hide_index=True)
        else:
            st.warning("Aucune image testee. Verifiez dataset/.")


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
        <p>Identifiez un lieu touristique marocain a partir d'une photo · Fiche historique generee par IA locale</p>
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
        seuil          = st.slider("Seuil de confiance", 0.0, 1.0, SEUIL_DEFAULT, 0.05,
                                    help="En dessous = Lieu non reconnu")
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
            st.success(f"{n_docs} images indexees")

        st.divider()
        if st.button("Reindexer le dataset", use_container_width=True):
            with st.spinner("Indexation..."):
                nb = reindexer(clip_model, collection, metadata)
            st.success(f"{nb} images reindexees !")
            st.rerun()

        st.divider()
        st.caption(f"CLIP: clip-ViT-B-32 | LLM: {modele_choisi}")
        st.caption("Hackathon Visual RAG Maroc")

    # ── Onglets ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Identification", "Lieux disponibles", "Statistiques"])

    # ── TAB 1 — Identification ─────────────────────────────────────────────────
    with tab1:
        if n_docs == 0:
            st.warning("Base vide. Cliquez Reindexer dans la sidebar.")
            st.stop()

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
            go = st.button("Identifier et generer la fiche", type="primary", use_container_width=True)
            if uploaded:
                st.image(uploaded, caption="Image selectionnee", use_container_width=True)

        with col_output:
            st.subheader("Resultats")

            if go and uploaded:
                image = Image.open(uploaded).convert("RGB")

                with st.spinner("Recherche vectorielle..."):
                    resultats = rechercher_lieu(image, clip_model, collection, n_results)

                if not resultats:
                    st.error("Aucun resultat.")
                    st.stop()

                st.markdown("**Lieux candidats :**")
                for r in resultats:
                    pct = max(0, min(100, int(r["score"]*100)))
                    st.progress(pct, text=f"#{r['rang']} {r['nom']} ({r['ville']}) - {r['score']:.4f}")

                st.divider()
                lieu = resultats[0]

                # Seuil de confiance
                if lieu["score"] < seuil:
                    st.error(f"Lieu non reconnu (score {lieu['score']:.2%} < seuil {seuil:.0%})")
                    st.info("Essayez une photo plus nette ou verifiez que le lieu est dans le dataset.")
                    st.stop()

                # Comparaison cote a cote
                st.markdown("**Comparaison : votre photo vs reference du dataset**")
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

                # Infos lieu
                c_nom, c_badge = st.columns([3,2])
                with c_nom:
                    st.markdown(f"**{lieu['nom']} — {lieu['ville']}**")
                    if lieu.get("epoque"):
                        st.caption(f"{lieu['epoque']}  |  {lieu.get('style','')}")
                with c_badge:
                    st.markdown(badge_confiance(lieu["score"], seuil), unsafe_allow_html=True)
                    st.caption(f"Score : {lieu['score']:.2%}")

                # Carte GPS
                if lieu.get("lat") and lieu["lat"] != 0:
                    st.map(pd.DataFrame([{"lat": lieu["lat"], "lon": lieu["lon"]}]), zoom=8)

                # Generation avec barre de progression en temps reel
                st.markdown("**Generation de la fiche...**")
                prog_bar = st.progress(0, text="En attente du LLM...")
                t_debut  = time.time()
                reponse  = [None]

                def call_llm():
                    reponse[0] = generate_response(lieu, question, modele_choisi, langue_choisie, timeout_llm)

                thread = threading.Thread(target=call_llm)
                thread.start()
                while thread.is_alive():
                    elapsed = time.time() - t_debut
                    pct = min(int((elapsed / timeout_llm) * 100), 95)
                    prog_bar.progress(pct, text=f"Generation en cours... {elapsed:.0f}s")
                    time.sleep(0.5)
                thread.join()
                elapsed_total = time.time() - t_debut
                prog_bar.progress(100, text=f"Genere en {elapsed_total:.1f}s")
                time.sleep(0.3)
                prog_bar.empty()

                reponse_finale = reponse[0] or "Erreur lors de la generation."
                st.markdown("**Fiche du guide touristique IA :**")
                st.markdown(f'<div class="fiche-box">{reponse_finale}</div>', unsafe_allow_html=True)

                # Actions
                st.divider()
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.download_button(
                        "Telecharger la fiche",
                        data=export_txt(lieu, reponse_finale),
                        file_name=f"fiche_{lieu['nom'].replace(' ','_')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with c2:
                    if st.button("Correct", use_container_width=True):
                        st.toast("Merci !", icon="OK")
                with c3:
                    if st.button("Incorrect", use_container_width=True):
                        st.toast("Retour note.", icon="NOTE")

                # Copier presse-papier
                st.text_area("Copier la fiche (Ctrl+A puis Ctrl+C)",
                              value=reponse_finale, height=100)

                # Historique
                st.session_state.historique.insert(0, {
                    "nom"  : lieu["nom"],
                    "ville": lieu["ville"],
                    "score": lieu["score"],
                    "heure": datetime.datetime.now().strftime("%H:%M:%S"),
                })
                st.session_state.historique = st.session_state.historique[:15]

            elif go and not uploaded:
                st.warning("Choisissez une image d'abord.")
            else:
                st.info("Uploadez une image et cliquez sur Identifier.")
                if st.session_state.historique:
                    st.markdown("**Dernieres recherches**")
                    for h in st.session_state.historique[:5]:
                        st.markdown(f"""
                        <div class="hist-item">
                            <strong>{h['nom']}</strong> — {h['ville']}
                            &nbsp;|&nbsp; {h['score']:.2%}
                            &nbsp;|&nbsp; {h['heure']}
                        </div>
                        """, unsafe_allow_html=True)

    # ── TAB 2 — Lieux disponibles ──────────────────────────────────────────────
    with tab2:
        page_accueil(metadata)

    # ── TAB 3 — Statistiques ───────────────────────────────────────────────────
    with tab3:
        page_statistiques(st.session_state.historique, metadata, clip_model, collection)


if __name__ == "__main__":
    main()