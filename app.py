
import io
from collections import Counter, defaultdict

import pandas as pd
import spacy
import streamlit as st


# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Analyseur lexical",
    layout="centered"
)

LEXIQUE_FILE = "Lexique383.tsv"


# =====================================================
# 2. CHARGEMENT DE LEXIQUE
# =====================================================

@st.cache_data
def load_lexique():

    lexique = pd.read_csv(
        LEXIQUE_FILE,
        sep="\t",
        encoding="utf-8"
    )

    required = [
        "ortho",
        "lemme",
        "freqlemlivres",
        "freqlemfilms2",
        "islem"
    ]

    for column in required:
        if column not in lexique.columns:
            raise ValueError(
                f"Colonne manquante : {column}"
            )

    for column in [
        "freqlemlivres",
        "freqlemfilms2",
        "islem"
    ]:
        lexique[column] = pd.to_numeric(
            lexique[column],
            errors="coerce"
        )

    lexique["ortho_norm"] = (
        lexique["ortho"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .str.replace("œ", "oe", regex=False)
    )

    lexique["lemme_norm"] = (
        lexique["lemme"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .str.replace("œ", "oe", regex=False)
    )

    database = lexique[
        (lexique["islem"] == 1)
        & lexique["freqlemlivres"].notna()
        & (lexique["freqlemlivres"] > 0)
    ].copy()

    database = (
        database.groupby("lemme_norm", as_index=False)
        .agg({
            "freqlemlivres": "max",
            "freqlemfilms2": "max"
        })
    )

    database = database.rename(
        columns={"lemme_norm": "lemme"}
    )

    database = database.sort_values(
        ["freqlemlivres", "lemme"],
        ascending=[False, True]
    ).reset_index(drop=True)

    database["rank"] = range(
        1,
        len(database) + 1
    )

    frequency_dict = (
        database.set_index("lemme")
        .to_dict("index")
    )

    forms = lexique[
        lexique["ortho_norm"].notna()
        & lexique["lemme_norm"].isin(frequency_dict)
    ]

    form_dict = defaultdict(set)

    for ortho, lemma in zip(
        forms["ortho_norm"],
        forms["lemme_norm"]
    ):
        form_dict[ortho].add(lemma)

    return frequency_dict, dict(form_dict), len(form_dict)


# =====================================================
# 3. MODÈLE SPACY
# =====================================================

@st.cache_resource
def load_spacy():

    return spacy.load("fr_core_news_lg")


# =====================================================
# 4. LEMMATISATION
# =====================================================

def analyse_text(TEXT, frequency_dict, form_dict, nlp):

    doc = nlp(TEXT)

    words = []

    for token in doc:

        if not token.is_alpha:
            continue

        surface = (
            token.text
            .casefold()
            .replace("œ", "oe")
        )

        spacy_lemma = (
            token.lemma_
            .casefold()
            .strip()
            .replace("œ", "oe")
        )

        candidates = form_dict.get(surface, set())

        if len(candidates) == 1:

            lemma = next(iter(candidates))
            method = "Lexique"

        elif len(candidates) > 1:

            if spacy_lemma in candidates:
                lemma = spacy_lemma
                method = "Lexique + spaCy"
            else:
                lemma = spacy_lemma
                method = "Repli spaCy"

        else:

            lemma = spacy_lemma
            method = "Repli spaCy"

        if not lemma:
            lemma = surface
            method = "Forme originale"

        words.append({
            "original": token.text,
            "lemma": lemma,
            "method": method
        })

    return words


# =====================================================
# 5. CALCUL DES RÉSULTATS
# =====================================================

def calculate_results(words, frequency_dict):

    counts = Counter(
        word["lemma"] for word in words
    )

    method_counts = Counter(
        word["method"] for word in words
    )

    words_by_lemma = defaultdict(list)

    for word in words:
        words_by_lemma[word["lemma"]].append(word)

    results = []

    for lemma, occurrences in counts.items():

        data = frequency_dict.get(lemma)

        original_forms = sorted({
            word["original"]
            for word in words_by_lemma[lemma]
        })

        methods = sorted({
            word["method"]
            for word in words_by_lemma[lemma]
        })

        results.append({

            "lemma": lemma,

            "original_forms": ", ".join(original_forms),

            "lemmatization_method": ", ".join(methods),

            "occurrences": occurrences,

            "frequency_rank": (
                data["rank"] if data else None
            ),

            "frequency_books": (
                data["freqlemlivres"] if data else None
            ),

            "frequency_films": (
                data["freqlemfilms2"] if data else None
            )
        })

    df = pd.DataFrame(results)

    matched = df.dropna(
        subset=["frequency_rank"]
    ).copy()

    if not matched.empty:

        # Moyenne non pondérée

        unweighted_average = (
            matched["frequency_rank"].mean()
        )

        # Moyenne pondérée

        weighted_average = (
            matched["frequency_rank"]
            * matched["occurrences"]
        ).sum() / matched["occurrences"].sum()

        # Médiane non pondérée

        unweighted_median = (
            matched["frequency_rank"].median()
        )

        # Médiane pondérée

        weighted_sorted = matched.sort_values(
            "frequency_rank"
        )

        cumulative = (
            weighted_sorted["occurrences"].cumsum()
        )

        total_occurrences = int(
            weighted_sorted["occurrences"].sum()
        )

        lower_position = (
            total_occurrences - 1
        ) // 2

        upper_position = (
            total_occurrences
        ) // 2

        lower_value = weighted_sorted.loc[
            cumulative.gt(lower_position),
            "frequency_rank"
        ].iloc[0]

        upper_value = weighted_sorted.loc[
            cumulative.gt(upper_position),
            "frequency_rank"
        ].iloc[0]

        weighted_median = (
            lower_value + upper_value
        ) / 2

    else:

        unweighted_average = None
        weighted_average = None
        unweighted_median = None
        weighted_median = None

    df = df.sort_values(
        "frequency_rank",
        ascending=False,
        na_position="first"
    ).reset_index(drop=True)

    return {
        "df": df,
        "matched": matched,
        "counts": counts,
        "method_counts": method_counts,
        "words": words,
        "unweighted_average": unweighted_average,
        "weighted_average": weighted_average,
        "unweighted_median": unweighted_median,
        "weighted_median": weighted_median
    }


# =====================================================
# 6. EXPORTATION CSV COMPLÈTE
# =====================================================

def create_csv(result, frequency_dict, form_count):

    df = result["df"]
    words = result["words"]
    counts = result["counts"]
    matched = result["matched"]
    method_counts = result["method_counts"]

    statistics = {

        "Nombre total d'occurrences": len(words),

        "Lemmes distincts": len(counts),

        "Lemmes trouvés dans Lexique": len(matched),

        "Lemmes absents de Lexique": (
            len(counts) - len(matched)
        ),

        "Rang moyen non pondéré": (
            result["unweighted_average"]
        ),

        "Rang moyen pondéré": (
            result["weighted_average"]
        ),

        "Rang médian non pondéré": (
            result["unweighted_median"]
        ),

        "Rang médian pondéré": (
            result["weighted_median"]
        ),

        "Lemmes Lexique chargés": len(frequency_dict),

        "Formes orthographiques chargées": form_count
    }

    records = []

    # STATISTIQUES GÉNÉRALES

    for name, value in statistics.items():

        records.append({
            "type_enregistrement": "STATISTIQUE",
            "catégorie": "Général",
            "nom": name,
            "valeur": value
        })

    # MÉTHODES DE LEMMATISATION

    for method, count in method_counts.items():

        records.append({

            "type_enregistrement": "STATISTIQUE",

            "catégorie": "Lemmatisation",

            "nom": method,

            "valeur": count,

            "pourcentage": (
                count / len(words) * 100
                if words else 0
            )
        })

    # DIAGNOSTICS DES MOTS COURANTS

    for word in [
        "je", "tu", "il", "elle", "nous",
        "vous", "être", "avoir", "le", "de"
    ]:

        data = frequency_dict.get(word)

        records.append({

            "type_enregistrement": "DIAGNOSTIC",

            "catégorie": "Mots courants",

            "nom": word,

            "valeur": (
                "TROUVÉ" if data else "INTROUVABLE"
            ),

            "rang_fréquence": (
                data["rank"] if data else None
            ),

            "fréquence_livres": (
                data["freqlemlivres"] if data else None
            ),

            "fréquence_films": (
                data["freqlemfilms2"] if data else None
            )
        })

    # VOCABULAIRE COMPLET

    for _, row in df.iterrows():

        records.append({

            "type_enregistrement": "LEMME",

            "catégorie": "Vocabulaire",

            "lemme": row["lemma"],

            "formes_originales": row["original_forms"],

            "méthode_lemmatisation": row["lemmatization_method"],

            "occurrences": row["occurrences"],

            "rang_fréquence": row["frequency_rank"],

            "fréquence_livres": row["frequency_books"],

            "fréquence_films": row["frequency_films"]
        })

    complete_csv = pd.DataFrame(records)

    return complete_csv.to_csv(
        index=False,
        encoding="utf-8-sig"
    ).encode("utf-8-sig")



# =====================================================
# 7. INTERFACE
# =====================================================

st.title("Analyseur lexical français")

st.write(
    "Cet outil analyse le vocabulaire d'un texte français "
    "à l'aide de Lexique 3.83 et du modèle linguistique "
    "spaCy. Il identifie les lemmes, mesure leur fréquence "
    "et calcule le rang moyen de fréquence du vocabulaire."
)

TEXT = st.text_area(
    "Texte",
    height=300,
    placeholder="",
    label_visibility="collapsed"
)

if st.button(
    "Analyser",
    type="primary",
    use_container_width=True
):

    if not TEXT.strip():

        st.warning("Veuillez saisir un texte.")

    else:

        try:

            frequency_dict, form_dict, form_count = (
                load_lexique()
            )

            nlp = load_spacy()

            with st.spinner("Analyse en cours..."):

                words = analyse_text(
                    TEXT,
                    frequency_dict,
                    form_dict,
                    nlp
                )

                result = calculate_results(
                    words,
                    frequency_dict
                )

                csv_bytes = create_csv(
                    result,
                    frequency_dict,
                    form_count
                )

            st.session_state["mean"] = (
                result["unweighted_average"]
            )

            st.session_state["csv"] = csv_bytes
            st.session_state["analysed"] = True

        except Exception as e:

            st.error(f"Erreur : {e}")


# =====================================================
# 8. RÉSULTAT
# =====================================================

if st.session_state.get("analysed", False):

    mean = st.session_state["mean"]

    st.divider()

    if mean is not None:

        st.metric(
            "Rang moyen non pondéré",
            f"{mean:,.0f}"
        )

    else:

        st.write("Aucune correspondance trouvée.")

    st.download_button(
        "Télécharger l'analyse complète",
        data=st.session_state["csv"],
        file_name="analyse_frequentielle_francaise.csv",
        mime="text/csv",
        use_container_width=True
    )

    # IMAGE — REMPLAÇABLE

    st.image(
        "image.jpg",
        use_container_width=True
    )
