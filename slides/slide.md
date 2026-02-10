---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  section {
    font-size: 22px;
    position: relative;
    padding-top: 80px;
  }
  section h1:first-of-type {
    position: absolute;
    top: 80px;
    left: 50px;
    right: 50px;
    margin: 0;
    font-size: 34px;
  }
  section.title {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding-top: 0;
  }
  section.title img {
    max-width: 150px;
    max-height: 150px;
    width: auto;
    height: auto;
    object-fit: contain;
  }
  section.title h1, section.title h2, section.title h3 {
    position: static;
    margin: 10px 0;
  }
  h2 {
    font-size: 26px;
  }
  h3 {
    font-size: 22px;
  }
  code {
    font-size: 16px;
  }
  pre {
    font-size: 14px;
  }
  ul, ol {
    font-size: 20px;
  }
  table {
    font-size: 18px;
  }
  .columns {
    display: flex;
    gap: 40px;
  }
  .columns > * {
    flex: 1;
  }
  section img {
    max-width: 70%;
    max-height: 70%;
    display: block;
    margin: 0 auto;
  }  a {
    color: #003366;
  }
  section::before {
    content: 'Daniele Molinari';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 32px;
    background: #4a90d9;
    color: white;
    font-size: 14px;
    font-weight: 500;
    display: flex;
    align-items: center;
    padding-left: 20px;
    box-sizing: border-box;
  }
  section.title::before {
    display: none;
  }
  
  

---

<!-- _class: title -->
<!-- _paginate: false -->

![w:150](img/unipr.png)

# Analisi di Reti Trofiche
## Utilizzo di tecniche non supervisionate per l'analisi di reti trofiche

<br>
<br>
<br>


**Daniele Molinari**
Progetto di Laboratorio di Intelligenza Artificiale

---

# Introduzione

Le **reti trofiche** sono grafi diretti pesati che modellano 
le relazioni predatore-preda in un ecosistema:
- **Nodi** — Specie
- **Archi** — Flusso di energia (preda → predatore)
- **Pesi** — Intensità dell'interazione

![bg 80% right:35% fit](img/rete_disegno.png)

<br>
<br>

**Domanda centrale:**
> *Cosa succede quando una specie si estingue? Come si propaga l'effetto nella rete?*

---

# Obiettivi del Progetto

Il progetto si articola in **due parti**:

1. **Analisi singola rete**: Simulare estinzioni a cascata e verificare se le comunità **compartimentalizzano** il danno
2. **Analisi Esplorativa**: Identificare pattern strutturali comuni tra 33 reti trofiche

<br>
<br>

**Dataset:** Web of Life — reti ecologiche reali da ecosistemi marini, fluviali e di lago

---

# Il Modello Energetico (Bellingeri et al.)

Una specie si estingue quando l'energia in ingresso scende sotto una **soglia critica**:

<br>

$${E_{\text{corrente}} \leq th \times E_{\text{iniziale}}}$$

<br>

Quindi, ad esempio:
- $th = 0.5$:  Estinzione se energia scende sotto il 50% 
- $th$ alto: Rete fragile
- $th$ basso: ete robusta 

---

# Indice di Robustezza $R_\alpha$

<br>
<br>

Una prima analisi di robustezza della rete è stata qella di quantificare quante **estinzioni primarie** sono necessarie per causare $\alpha$% di estinzioni totali:

$$R_\alpha = \frac{E_{\text{primarie}}}{S}$$

<br>


![w:500](img/energetic_r50.png)

---

# Community Detection

- **Ipotesi:** Le reti trofiche hanno struttura modulare.  
Un'estinzione dovrebbe colpire principalmente la propria comunità.

- **Metodo:** Algoritmo di Louvain (massimizza la modularità)

- Sulla rete **FW_008** (Caribbean Marine Food Web):
    - 249 specie, 3313 interazioni
    - Modularità: 0.48
    - 7 comunità identificate

![bg right:44% fit](img/rete_trofica.png)

---

# Visualizzazione delle Comunità

<br>
<br>

![w:650](img/community_graph.png)

---

# Propagazione delle Estinzioni

<br>
<br>

Nella seguente **heatmap** è possibile vedere come sono distribuite le estinzioni secondarie causate dalle estinzioni primarie.

![w:550](img/heatmap_community_estinzioni.png)


---

# Parte 2: Analisi Esplorativa

<br>

**Obiettivo:** Identificare pattern comuni tra **33 reti trofiche** da ambienti diversi. Per farlo verifichiamo la formazione di cluster naturali.



**Feature estratte per ogni rete:**
<center>

<br>

| Feature | Cosa misura |
|---------|-------------|
| `Size` | Numero di nodi nella rete (specie) |
| `connectance` | Densità dei link ($L/S^2$) |
| `modularity` | Struttura a comunità |
| `avg_energy_flow` | Intensità media delle interazioni |
| `auc_robustness` | Fragilità globale (area sotto curva impatto) |

---

# Pipeline di Clustering

1. **StandardScaler** — normalizzazione delle feature
2. **UMAP** — riduzione dimensionale (preserva struttura locale)
3. **K-Means** — clustering con scelta automatica di K

**Scelta di K:** Silhouette Score → $K_{opt} = 3$

![bg right:44% fit](img/umap_cluster_results.png)

---

# Biomi vs Cluster

<br>
<br>

**Ipotesi:** I cluster corrispondono ai biomi (mare, fiume, lago)?

<br>

![w:900](img/biomi_cluster_comparison.png)

<br>

**Risultato:** No. Reti di ambienti diversi finiscono nello stesso cluster.

---

# Interpretazione dei Cluster

<br>
<br>

- **Cluster 0**: Bassa connettività, modularità media e Robusto
- **Cluster 1**: Alta connettività, bassa modularità e Fragile
- **Cluster 2**: Alta modularità, estinzioni localizzate e Intermedio

<br>

![w:900](img/boxplot_clusters.png)

---

# Conclusioni

I risultati indicano che reti con simili livelli di modularità e connettività tendono a condividere anche livelli simili di fragilità, evidenziando un pattern strutturale coerente tra organizzazione topologica e robustezza della rete.

<br>

**Implicazione:** La protezione delle specie hub e il monitoraggio della struttura topologica sono priorità per la conservazione.

![bg 85% right:45% fit](img/rete_disegno_finale.jpg)

---

<!-- _class: title -->
<!-- _paginate: false -->

![w:150](img/unipr.png)

# GRAZIE PER L'ATTENZIONE

<br>

**Repository**: github.com/osryde/trophic_graph_analysis
**Demo**: Vediamo un esempio di rete poco connessa
