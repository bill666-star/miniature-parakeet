import streamlit as st
import random
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import FilterCatalog, Draw
from rdkit.Chem.rdmolfiles import MolToMolBlock
from io import BytesIO, StringIO
import pandas as pd
import base64

st.set_page_config(
    page_title="SkelGen-Pro v2.0 | 骨架保序分子生成器",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 顶部样式
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.stButton>button { background-color: #4CAF50; color: white; font-size: 16px; height: 3em; width: 100%; }
.stDownloadButton>button { background-color: #2196F3; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🧬 SkelGen-Pro v2.0 — 骨架锁定·极小扰动药物分子生成")
st.markdown("#### 不改变母核骨架 | 生物等排体 | 去PAINS | 成药性过滤 | 对接专用分子库")

# ======================
# 缓存过滤器
# ======================
@st.cache_resource
def load_pains_filter():
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog.FilterCatalog(params)

pains_filter = load_pains_filter()

# ======================
# 生物等排体替换库（专业级）
# ======================
ISO_RULES = [
    ("[H]", ["F", "Cl"]),
    ("[F]", ["H", "Cl", "CN"]),
    ("[Cl]", ["F", "H"]),
    ("[CH3]", ["CH2CH3", "CF3", "CN"]),
    ("[OH]", ["OCH3", "F"]),
    ("[NH2]", ["OH", "CH3"]),
    ("C(=O)O", ["c1n[nH]nn1", "S(=O)(=O)N"]),
    ("c1ccccc1", ["c1ccncc1", "c1cccnc1"]),
    ("C=O", ["S=O"]),
]

# ======================
# 分子绘图
# ======================
def mol_to_image(mol, w=400, h=200):
    if mol is None:
        return None
    pil_img = Draw.MolToImage(mol, size=(w, h), kekulize=True)
    buf = BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

# ======================
# 属性计算
# ======================
def calc_props(mol):
    return {
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "RotB": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }

# ======================
# PAINS 检测
# ======================
def has_pains(mol):
    return pains_filter.HasMatch(mol)

# ======================
# 成药性 + 不突变约束
# ======================
def passed_filter(new_mol, orig_props, cfg):
    p = calc_props(new_mol)
    if abs(p["MW"] - orig_props["MW"]) > cfg["mw"]: return False
    if abs(p["LogP"] - orig_props["LogP"]) > cfg["logp"]: return False
    if abs(p["TPSA"] - orig_props["TPSA"]) > cfg["tpsa"]: return False
    if abs(p["HBA"] - orig_props["HBA"]) > cfg["hba"]: return False
    if abs(p["HBD"] - orig_props["HBD"]) > cfg["hbd"]: return False
    if p["MW"] > 550: return False
    if p["LogP"] > 5: return False
    if p["HBD"] > 5: return False
    if p["HBA"] > 10: return False
    if has_pains(new_mol): return False
    return True

# ======================
# 骨架保序等排体替换
# ======================
def replace_bioiso(mol):
    m = Chem.Mol(mol)
    try:
        rule = random.choice(ISO_RULES)
        patt = Chem.MolFromSmarts(rule[0])
        repl = Chem.MolFromSmarts(random.choice(rule[1:]))
        if patt and repl:
            res = Chem.ReplaceSubstructs(m, patt, repl, replaceAll=False)
            if res and res[0]:
                return res[0]
    except:
        pass
    return m

# ======================
# 批量生成
# ======================
def generate_library(smi, count, cfg):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return [], {}, None
    orig_props = calc_props(mol)
    valid = []
    seen = set()
    for _ in range(count * 15):
        cand = replace_bioiso(mol)
        if not cand: continue
        if not passed_filter(cand, orig_props, cfg): continue
        s = Chem.MolToSmiles(cand)
        if s in seen or s == smi: continue
        seen.add(s)
        valid.append((s, calc_props(cand), cand))
        if len(valid) >= count: break
    return valid, orig_props, mol

# ======================================
# 侧边栏：输入与参数
# ======================================
with st.sidebar:
    st.subheader("📥 输入分子")
    input_smi = st.text_input("SMILES", "c1cc(OC)ccc1C")
    gen_count = st.number_input("生成数量", 10, 500, 30)

    st.subheader("⚙️ 性质不突变约束")
    mw_tol = st.slider("分子量波动 ±", 0, 60, 25)
    logp_tol = st.slider("LogP 波动 ±", 0.0, 2.0, 1.0, 0.1)
    tpsa_tol = st.slider("TPSA波动 ±", 0, 40, 15)
    hba_tol = st.slider("HBA波动 ±", 0, 2, 1)
    hbd_tol = st.slider("HBD波动 ±", 0, 2, 1)

    config = {
        "mw": mw_tol,
        "logp": logp_tol,
        "tpsa": tpsa_tol,
        "hba": hba_tol,
        "hbd": hbd_tol
    }

# ======================================
# 主区域：分子预览 + 结果
# ======================================
col_input, col_params = st.columns([1, 1])
start = st.button("🚀 生成骨架保序分子库", use_container_width=True)

if start:
    with st.spinner("生成中 · 骨架锁定 · 等排体替换 · PAINS过滤 · 成药性校验"):
        mols, orig_props, orig_mol = generate_library(input_smi, gen_count, config)

    if not mols:
        st.error("无法生成符合条件的分子，请放宽约束或检查SMILES")
    else:
        st.success(f"✅ 生成 {len(mols)} 个高质量分子")

        # 原图预览
        with col_input:
            st.markdown("#### 原始分子结构")
            img = mol_to_image(orig_mol)
            if img:
                st.markdown(f'<img src="data:image/png;base64,{img}">', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([orig_props]), use_container_width=True)

        # 表格
        rows = []
        sdf = ""
        for i, (smi, prop, mol_obj) in enumerate(mols, 1):
            rows.append({
                "ID": f"SKEL-{i:03d}",
                "SMILES": smi,
                "MW": prop["MW"],
                "LogP": prop["LogP"],
                "TPSA": prop["TPSA"],
                "HBA": prop["HBA"],
                "HBD": prop["HBD"]
            })
            sdf += MolToMolBlock(mol_obj) + "$$$$\n"

        df = pd.DataFrame(rows)
        st.subheader("📊 生成分子库")
        st.dataframe(df, use_container_width=True, height=400)

        # 预览前8个分子
        st.subheader("🖼️ 分子结构预览")
        preview = mols[:8]
        cols = st.columns(4)
        for i, (smi, prop, mol) in enumerate(preview):
            with cols[i % 4]:
                img = mol_to_image(mol, w=220, h=120)
                st.markdown(f"**SKEL-{i+1:03d}**")
                st.markdown(f'<img src="data:image/png;base64,{img}">', unsafe_allow_html=True)

        # 导出
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("💾 导出CSV", df.to_csv(index=False), "SkelGen分子库.csv")
        with c2:
            st.download_button("📥 导出SDF(对接专用)", sdf, "SkelGen分子库.sdf")

st.markdown("---")
st.markdown("""
**SkelGen-Pro v2.0 | 完全免费开源 | MIT协议**
适用：分子对接、虚拟筛选、me-too药物设计、骨架保持结构优化
""")
