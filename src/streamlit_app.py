"""
🌿 Extinction Cascade Simulator - Streamlit Version
Simula estinzioni primarie e osserva l'effetto cascata nelle reti trofiche
"""

import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import community as community_louvain
import plotly.graph_objects as go
import plotly.express as px
import random

# Page config
st.set_page_config(
    page_title="Extinction Simulator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Apple-inspired minimal design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: #fafafa;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e5e7;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #1d1d1f;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        font-size: 2.8rem;
        font-weight: 600;
        color: #1d1d1f;
        letter-spacing: -0.02em;
    }
    
    .main-header span {
        background: linear-gradient(135deg, #34c759 0%, #30d158 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
        color: #1d1d1f;
    }
    
    [data-testid="stMetricLabel"] {
        color: #86868b;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(180deg, #1d1d1f 0%, #000000 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(180deg, #34c759 0%, #28a745 100%);
    }
    
    .stButton > button[kind="secondary"] {
        background: #f5f5f7;
        color: #1d1d1f;
    }
    
    /* Inputs */
    .stSlider > div > div {
        background: #e5e5e7;
    }
    
    .stSlider > div > div > div {
        background: #34c759;
    }
    
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: #ffffff;
        border: 1px solid #e5e5e7;
        border-radius: 12px;
    }
    
    .stNumberInput > div > div > input {
        background: #ffffff;
        border: 1px solid #e5e5e7;
        border-radius: 12px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e5e5e7;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #f5f5f7;
        border-radius: 12px;
        padding: 4px;
        gap: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #86868b;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: #ffffff;
        color: #1d1d1f;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* DataFrame */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e5e5e7;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: #e5e5e7;
        margin: 1.5rem 0;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 2px dashed #e5e5e7;
        border-radius: 16px;
        padding: 1rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #34c759;
        background: #f0fff4;
    }
    
    /* Success/Warning/Info boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #34c759 transparent transparent transparent;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom card style */
    .apple-card {
        background: #ffffff;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06);
        border: 1px solid #e5e5e7;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #86868b;
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: -0.5rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============ HELPER FUNCTIONS ============

def community_layout(G, partition, scale=3, seed=42):
    """
    Create a layout that positions nodes of the same community close together.
    Communities are arranged in a circle, with nodes spread within each community.
    """
    np.random.seed(seed)
    
    # Group nodes by community
    communities = {}
    for node, comm in partition.items():
        if comm not in communities:
            communities[comm] = []
        communities[comm].append(node)
    
    # Place community centroids in a circle
    n_comm = len(communities)
    comm_centers = {}
    for i, comm in enumerate(sorted(communities.keys())):
        angle = 2 * np.pi * i / n_comm
        comm_centers[comm] = (scale * np.cos(angle), scale * np.sin(angle))
    
    # For each community, position nodes around the centroid
    pos = {}
    for comm, nodes in communities.items():
        cx, cy = comm_centers[comm]
        # Create subgraph for the community
        subgraph = G.subgraph(nodes)
        # Local layout for community nodes
        if len(nodes) > 1:
            local_pos = nx.spring_layout(subgraph, k=0.5, seed=seed, iterations=50)
        else:
            local_pos = {nodes[0]: (0, 0)}
        
        # Translate local positions to community centroid
        for node, (x, y) in local_pos.items():
            pos[node] = (cx + x * 0.8, cy + y * 0.8)
    
    return pos, comm_centers, communities


@st.cache_data
def load_network(uploaded_file):
    """Load and process network from CSV"""
    df = pd.read_csv(uploaded_file, index_col=0)
    all_species = sorted(list(set(df.index) | set(df.columns)))
    df_square = df.reindex(index=all_species, columns=all_species, fill_value=0)
    return df_square


def create_graph(df_square):
    """Create NetworkX graph from adjacency matrix"""
    G = nx.from_pandas_adjacency(df_square, create_using=nx.DiGraph)
    return G


def get_network_stats(G, partition):
    """Calculate network statistics"""
    G_undirected = G.to_undirected()
    modularity = community_louvain.modularity(partition, G_undirected, weight='weight')
    
    n_nodes = len(G.nodes())
    n_edges = len(G.edges())
    basal = [n for n in G.nodes() if G.in_degree(n) == 0]
    apex = [n for n in G.nodes() if G.out_degree(n) == 0]
    connectance = n_edges / (n_nodes ** 2) if n_nodes > 0 else 0
    
    return {
        'n_species': n_nodes,
        'n_interactions': n_edges,
        'n_basal': len(basal),
        'n_apex': len(apex),
        'basal_species': basal,
        'apex_species': apex,
        'connectance': round(connectance, 4),
        'modularity': round(modularity, 4),
        'n_communities': len(set(partition.values()))
    }


def simulate_extinction_cascade(graph, primary_nodes, th):
    """Simulate extinction cascade"""
    temp_graph = graph.copy()
    initial_energy = dict(graph.in_degree(weight='weight'))
    
    cascade_steps = []
    all_extinct = set()
    
    # Primary extinctions
    primary_extinct = set()
    for node in primary_nodes:
        if node in temp_graph:
            temp_graph.remove_node(node)
            primary_extinct.add(node)
            all_extinct.add(node)
    
    cascade_steps.append({
        'step': 0,
        'type': 'primary',
        'extinct_nodes': list(primary_extinct),
        'remaining': len(temp_graph.nodes())
    })
    
    # Secondary cascade
    step_num = 1
    new_extinctions = True
    
    while new_extinctions:
        new_extinctions = False
        step_extinct = []
        
        for n in list(temp_graph.nodes()):
            if initial_energy[n] > 0:
                current_energy = temp_graph.in_degree(n, weight='weight')
                if current_energy <= th * initial_energy[n]:
                    step_extinct.append(n)
                    new_extinctions = True
        
        if step_extinct:
            for n in step_extinct:
                all_extinct.add(n)
                temp_graph.remove_node(n)
            
            cascade_steps.append({
                'step': step_num,
                'type': 'secondary',
                'extinct_nodes': step_extinct,
                'remaining': len(temp_graph.nodes())
            })
            step_num += 1
    
    return {
        'cascade_steps': cascade_steps,
        'total_extinct': len(all_extinct),
        'primary_extinct': len(primary_extinct),
        'secondary_extinct': len(all_extinct) - len(primary_extinct),
        'survivors': list(temp_graph.nodes()),
        'all_extinct': list(all_extinct)
    }


def calculate_keystone_species(G, th=0.5, top_n=10):
    """Find keystone species"""
    results = []
    for node in G.nodes():
        if G.out_degree(node) > 0:
            cascade = simulate_extinction_cascade(G, [node], th)
            results.append({
                'species': node,
                'secondary_extinctions': cascade['secondary_extinct'],
                'total': cascade['total_extinct']
            })
    results.sort(key=lambda x: x['secondary_extinctions'], reverse=True)
    return results[:top_n]


def create_network_plot(G, partition, extinct_nodes=None):
    """Create interactive Plotly network visualization with community compartments"""
    if extinct_nodes is None:
        extinct_nodes = set()
    else:
        extinct_nodes = set(extinct_nodes)
    
    # Use community layout
    pos, comm_centers, communities = community_layout(G, partition, scale=4, seed=42)
    
    # Color palette for communities - Apple inspired
    colors = ['#007AFF', '#5856D6', '#AF52DE', '#FF2D55', '#FF9500', '#FFCC00', '#34C759', '#00C7BE']
    
    fig = go.Figure()
    
    # Add community background circles/zones
    for comm_id, (cx, cy) in comm_centers.items():
        comm_nodes = communities[comm_id]
        if len(comm_nodes) > 0:
            # Calculate radius based on node spread
            node_positions = [pos[n] for n in comm_nodes if n in pos]
            if node_positions:
                max_dist = max(np.sqrt((x - cx)**2 + (y - cy)**2) for x, y in node_positions)
                radius = max(max_dist + 0.3, 0.5)
            else:
                radius = 0.5
            
            # Add circle for community zone
            theta = np.linspace(0, 2 * np.pi, 50)
            circle_x = cx + radius * np.cos(theta)
            circle_y = cy + radius * np.sin(theta)
            
            color = colors[comm_id % len(colors)]
            fig.add_trace(go.Scatter(
                x=circle_x.tolist(), y=circle_y.tolist(),
                fill='toself',
                fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.08)',
                line=dict(color=color, width=2, dash='dot'),
                mode='lines',
                hoverinfo='skip',
                showlegend=False
            ))
            
            # Add community label
            fig.add_annotation(
                x=cx, y=cy + radius + 0.2,
                text=f"<b>Comunità {comm_id}</b><br><span style='font-size:10px'>({len(comm_nodes)} specie)</span>",
                showarrow=False,
                font=dict(size=12, color=color),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor=color,
                borderwidth=1,
                borderpad=4
            )
    
    # Create edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        if u not in extinct_nodes and v not in extinct_nodes:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#86868b'),
        hoverinfo='none',
        mode='lines',
        opacity=0.25,
        showlegend=False
    ))
    
    # Create nodes
    node_x, node_y, node_colors, node_sizes, node_text = [], [], [], [], []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        comm = partition.get(node, 0)
        
        # Determine color
        if node in extinct_nodes:
            node_colors.append('#FF3B30')  # Apple Red
        elif in_deg == 0:  # Basal
            node_colors.append('#34C759')  # Apple Green
        elif out_deg == 0:  # Apex
            node_colors.append('#AF52DE')  # Apple Purple
        else:
            node_colors.append(colors[comm % len(colors)])
        
        node_sizes.append(12 + np.sqrt(in_deg + out_deg) * 3)
        status = " (ESTINTO)" if node in extinct_nodes else ""
        node_text.append(f"<b>{node}</b>{status}<br>Comunità: {comm}<br>In: {in_deg}, Out: {out_deg}")
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1.5, color='white'),
            opacity=0.95
        ),
        showlegend=False
    ))
    
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=20),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#fafafa',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='y'),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=580
    )
    
    return fig


def get_extinction_by_community(extinct_nodes, partition):
    """Get extinction count by community"""
    community_extinctions = {}
    for node in extinct_nodes:
        comm = partition.get(node, -1)
        if comm not in community_extinctions:
            community_extinctions[comm] = []
        community_extinctions[comm].append(node)
    return community_extinctions


def calculate_R_alpha(graph, th, alpha=0.5, strategy='random', simulations=10):
    """
    Calcola l'indice R_alpha per una data soglia th e una strategia di rimozione.
    R_alpha = frazione di estinzioni primarie necessarie per causare alpha% di estinzioni totali.
    
    Args:
        graph: NetworkX graph
        th: Soglia energetica di estinzione
        alpha: Percentuale di estinzioni totali da raggiungere (0.0-1.0)
        strategy: 'random' o 'most' (più connessi per primi)
        simulations: Numero di simulazioni per media (usato solo con strategy='random')
    
    Returns:
        R_alpha medio
    """
    S = len(graph.nodes())
    target = alpha * S
    
    r_alpha_values = []
    n_simulations = simulations if strategy == 'random' else 1
    
    for _ in range(n_simulations):
        temp_graph = graph.copy()
        initial_in_degree = {n: graph.in_degree(n, weight='weight') for n in graph.nodes()}
        
        # Scegliamo l'ordine di rimozione
        nodes_to_remove = list(graph.nodes())
        if strategy == 'most':
            # Ordine decrescente per numero di link (in + out)
            nodes_to_remove = sorted(nodes_to_remove, key=lambda x: graph.degree(x), reverse=True)
        else:
            random.shuffle(nodes_to_remove)
        
        extinct_total = set()
        primary_count = 0
        
        for p_node in nodes_to_remove:
            if p_node in extinct_total:
                continue
            
            # 1. Estinzione Primaria
            primary_count += 1
            extinct_total.add(p_node)
            if p_node in temp_graph:
                temp_graph.remove_node(p_node)
            
            # 2. Cascata di Estinzioni Secondarie (Logica Energetica)
            new_extinctions = True
            while new_extinctions:
                new_extinctions = False
                to_remove_secondary = []
                for node in temp_graph.nodes():
                    if initial_in_degree[node] > 0:
                        current_energy = temp_graph.in_degree(node, weight='weight')
                        # Se l'energia scende sotto la soglia th
                        if current_energy <= th * initial_in_degree[node]:
                            to_remove_secondary.append(node)
                            new_extinctions = True
                
                for node in to_remove_secondary:
                    extinct_total.add(node)
                    temp_graph.remove_node(node)
            
            # Controlliamo se abbiamo raggiunto alpha%
            if len(extinct_total) >= target:
                break
        
        r_alpha_values.append(primary_count / S)
    
    return sum(r_alpha_values) / len(r_alpha_values)


def create_robustness_chart(robustness_data):
    """Create robustness curve chart with R_alpha"""
    df = pd.DataFrame(robustness_data)
    
    # Calcola range y dinamico (come nel notebook)
    all_values = list(df['r_alpha_random']) + list(df['r_alpha_most'])
    y_min = min(all_values)
    y_max = max(all_values)
    margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.05
    
    fig = go.Figure()
    
    # Curva Random
    fig.add_trace(go.Scatter(
        x=df['threshold'],
        y=df['r_alpha_random'],
        mode='lines+markers',
        name='Random',
        line=dict(color='#007AFF', width=3),
        marker=dict(size=10, color='#007AFF')
    ))
    
    # Curva Most Connected
    fig.add_trace(go.Scatter(
        x=df['threshold'],
        y=df['r_alpha_most'],
        mode='lines+markers',
        name='Most Connected',
        line=dict(color='#FF3B30', width=3),
        marker=dict(size=10, color='#FF3B30')
    ))
    
    fig.update_layout(
        xaxis_title='Soglia Energetica (th)',
        yaxis_title='Robustezza (R<sub>α</sub>)',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#fafafa',
        font=dict(color='#1d1d1f'),
        xaxis=dict(gridcolor='#e5e5e7'),
        yaxis=dict(gridcolor='#e5e5e7', range=[max(0, y_min - margin), min(1, y_max + margin)]),
        height=350,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )
    
    return fig


# ============ MAIN APP ============

def main():
    # Header
    st.markdown('<h1 class="main-header"><span>Extinction</span> Simulator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Simula estinzioni e osserva l\'effetto cascata nelle reti trofiche</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📂 Carica Rete")
        uploaded_file = st.file_uploader(
            "Seleziona CSV",
            type=['csv'],
            help="File CSV con matrice di adiacenza",
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Parametri")
        
        threshold = st.slider(
            "Soglia di Estinzione (th)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Le specie si estinguono quando l'energia scende sotto th × energia iniziale"
        )
        
        st.markdown("---")
        st.markdown("### 🎨 Legenda")
        st.caption("🟢 Autotrofi · 🔵 Intermedi")
        st.caption("🟣 Apex · 🔴 Estinti")
    
    # Main content
    if uploaded_file is not None:
        # Load data
        df_square = load_network(uploaded_file)
        G = create_graph(df_square)
        partition = community_louvain.best_partition(G.to_undirected(), weight='weight', random_state=42)
        stats = get_network_stats(G, partition)
        
        # Initialize session state
        if 'extinct_nodes' not in st.session_state:
            st.session_state.extinct_nodes = []
        if 'cascade_result' not in st.session_state:
            st.session_state.cascade_result = None
        
        # Stats row
        st.markdown("#### Statistiche")
        cols = st.columns(6)
        metrics = [
            ("Specie", stats['n_species']),
            ("Link", stats['n_interactions']),
            ("Connettività", stats['connectance']),
            ("Modularità", stats['modularity']),
            ("Autotrofi", stats['n_basal']),
            ("Apex", stats['n_apex'])
        ]
        for col, (label, value) in zip(cols, metrics):
            col.metric(label, value)
        
        st.markdown("---")
        
        # Two columns: Controls + Network
        left_col, right_col = st.columns([1, 2])
        
        with left_col:
            st.markdown("#### Simulazione")
            
            # Tabs for manual vs random
            tab1, tab2 = st.tabs(["🎯 Manuale", "🎲 Random"])
            
            with tab1:
                eligible_species = [n for n in G.nodes() if G.out_degree(n) > 0]
                selected_species = st.multiselect(
                    "Seleziona specie da eliminare",
                    options=eligible_species,
                    help="Le specie apex (senza predatori) sono escluse"
                )
                
                if st.button("💀 Simula Estinzione", type="primary", use_container_width=True):
                    if selected_species:
                        result = simulate_extinction_cascade(G, selected_species, threshold)
                        st.session_state.extinct_nodes = result['all_extinct']
                        st.session_state.cascade_result = result
                        st.rerun()
                    else:
                        st.warning("Seleziona almeno una specie")
            
            with tab2:
                n_random = st.number_input("Numero estinzioni random", min_value=1, max_value=20, value=3)
                seed = st.number_input("Seed (0 = random)", min_value=0, value=0)
                
                if st.button("🎲 Estinzione Casuale", type="secondary", use_container_width=True):
                    if seed > 0:
                        random.seed(seed)
                    eligible = [n for n in G.nodes() if G.out_degree(n) > 0]
                    random_species = random.sample(eligible, min(n_random, len(eligible)))
                    result = simulate_extinction_cascade(G, random_species, threshold)
                    st.session_state.extinct_nodes = result['all_extinct']
                    st.session_state.cascade_result = result
                    st.info(f"Specie selezionate: {', '.join(random_species)}")
                    st.rerun()
            
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.extinct_nodes = []
                st.session_state.cascade_result = None
                st.rerun()
            
            # Cascade result
            if st.session_state.cascade_result:
                result = st.session_state.cascade_result
                st.markdown("---")
                st.markdown("#### Risultato")
                
                col1, col2 = st.columns(2)
                col1.metric("Primarie", result['primary_extinct'])
                col2.metric("Secondarie", result['secondary_extinct'])
                
                st.metric("Sopravvissuti", len(result['survivors']))
                
                # Community extinction statistics
                if result['all_extinct']:
                    comm_ext = get_extinction_by_community(result['all_extinct'], partition)
                    st.markdown("##### Estinzioni per Comunità")
                    
                    # Color palette
                    colors = ['#007AFF', '#5856D6', '#AF52DE', '#FF2D55', '#FF9500', '#FFCC00', '#34C759', '#00C7BE']
                    
                    for comm_id in sorted(comm_ext.keys()):
                        nodes_in_comm = comm_ext[comm_id]
                        color = colors[comm_id % len(colors)]
                        st.markdown(
                            f"<div style='padding: 8px 12px; margin: 4px 0; border-radius: 8px; "
                            f"background: linear-gradient(90deg, {color}22, transparent); "
                            f"border-left: 3px solid {color};'>"
                            f"<b style='color: {color}'>C{comm_id}</b>: {len(nodes_in_comm)} estinte"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                
                # Cascade timeline
                with st.expander("Timeline", expanded=False):
                    for step in result['cascade_steps']:
                        icon = "💀" if step['type'] == 'primary' else "🔥"
                        step_type = "Primaria" if step['type'] == 'primary' else f"Step {step['step']}"
                        species_list = ', '.join(step['extinct_nodes'][:3])
                        if len(step['extinct_nodes']) > 3:
                            species_list += f" (+{len(step['extinct_nodes'])-3})"
                        st.markdown(f"**{icon} {step_type}**: {species_list}")
        
        with right_col:
            st.markdown("#### Network")
            fig = create_network_plot(G, partition, st.session_state.extinct_nodes)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Bottom section: Keystone + Robustness
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Specie Keystone")
            if st.button("🔍 Analizza", use_container_width=True):
                with st.spinner("Analizzando..."):
                    keystone = calculate_keystone_species(G, threshold, top_n=8)
                    df_keystone = pd.DataFrame(keystone)
                    df_keystone.columns = ['Specie', 'Estinzioni Secondarie', 'Totale']
                    df_keystone.index = range(1, len(df_keystone) + 1)
                    df_keystone.index.name = 'Rank'
                    st.dataframe(df_keystone, use_container_width=True)
        
        with col2:
            st.markdown("#### Curva di Robustezza (R<sub>α</sub>)", unsafe_allow_html=True)
            
            # Slider per alpha
            alpha = st.slider(
                "α (% rete da estinguere)",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Percentuale di rete che deve estinguersi per calcolare R_α. Es: α=0.5 → R₅₀"
            )
            
            if st.button("📊 Calcola R_α", use_container_width=True):
                with st.spinner(f"Calcolando R_{int(alpha*100)}..."):
                    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    robustness_data = []
                    
                    for th in thresholds:
                        r_random = calculate_R_alpha(G, th, alpha=alpha, strategy='random', simulations=5)
                        r_most = calculate_R_alpha(G, th, alpha=alpha, strategy='most')
                        robustness_data.append({
                            'threshold': th,
                            'r_alpha_random': round(r_random, 3),
                            'r_alpha_most': round(r_most, 3)
                        })
                    
                    fig = create_robustness_chart(robustness_data)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostra interpretazione
                    st.caption(f"R_α = frazione di estinzioni primarie per causare {int(alpha*100)}% di estinzioni. Valori alti = rete robusta.")
    
    else:
        # Welcome screen - Apple style
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; background: #ffffff; border-radius: 24px; margin: 2rem auto; max-width: 600px; box-shadow: 0 4px 24px rgba(0,0,0,0.06); border: 1px solid #e5e5e7;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🧬</div>
            <h2 style="color: #1d1d1f; font-weight: 600; font-size: 1.8rem; margin-bottom: 0.5rem;">Inizia qui</h2>
            <p style="color: #86868b; font-size: 1.1rem; line-height: 1.6;">
                Carica un file CSV con la matrice di adiacenza<br>della tua rete trofica.
            </p>
            <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 2rem; color: #1d1d1f;">
                <div>📊<br><span style="color: #86868b; font-size: 0.85rem;">Centinaia di specie</span></div>
                <div>🔬<br><span style="color: #86868b; font-size: 0.85rem;">Simulazioni</span></div>
                <div>📈<br><span style="color: #86868b; font-size: 0.85rem;">Analisi avanzate</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Example format
        with st.expander("📖 Formato file CSV"):
            st.markdown("""
            Il file deve essere una **matrice di adiacenza** dove:
            - **Righe** = Prede (chi viene mangiato)
            - **Colonne** = Predatori (chi mangia)
            - **Valori** = Peso dell'interazione (flusso energetico)
            """)
            
            example_df = pd.DataFrame({
                'Predatore1': [0.5, 0.2, 0.0],
                'Predatore2': [0.3, 0.0, 0.6],
                'Predatore3': [0.0, 0.4, 0.1]
            }, index=['Preda1', 'Preda2', 'Preda3'])
            st.dataframe(example_df)


if __name__ == "__main__":
    main()
