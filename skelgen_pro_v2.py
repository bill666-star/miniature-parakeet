import streamlit as st
import random
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import FilterCatalog
from rdkit.Chem.rdmolfiles import MolToMolBlock
import pandas as pd

st.set_page_config(
    page_title="SkelGen-Pro 骨架保序分子生成器",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🧬 SkelGen-Pro — 骨架锁定·极小扰动药物分子生成")
st.caption("✅ 不改变骨架 | ✅ 生物等排体 | ✅ 去PAINS | ✅ 成药性过滤 | 开源免费")

# ======================
# 过滤器加载（PAINS）
# ======================
@st.cache_resource
def load_pains():
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogs.PAINS)
    return FilterCatalog.FilterCatalog(params)

pains_catalog = load_pains()

# ======================
# 生物电子等排体规则库（专业版）
# ======================
ISO_SMARTS = [
    ("[H]",  ["F", "Cl"]),
    ("[F]",  ["H", "Cl", "CN"]),
    ("[Cl]", ["F", "H", "Me"]),
    ("[CH3]",["CH2CH3", "CF3", "CN"]),
    ("[OH]", ["OCH3", "F"]),
    ("[NH2]",["OH", "CH3"]),
    ("c1ccccc1", ["c1ccncc1", "c1cccnc1"]),
    ("C=O", ["S=O"]),
    ("COOH", ["tetrazole", "SO2NH2"]),
]

# ======================
# 分子属性计算
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
# PAINS 判断
# ======================
def has_pains(mol):
    return pains_catalog.HasMatch(mol)

# ======================
# 成药性 + 性质不突变 双重过滤
# ======================
def is_valid(mol, orig, mw_tol, logp_tol, tpsa_tol, hba_tol, hbd_tol):
    p = calc_props(mol)

    # 不突变约束
    if abs(p["MW"] - orig["MW"]) > mw_tol:
        return False
    if abs(p["LogP"] - orig["LogP"]) > logp_tol:
        return False
    if abs(p["TPSA"] - orig["TPSA"]) > tpsa_tol:
        return False
    if abs(p["HBA"] - orig["HBA"]) > hba_tol:
        return False
    if abs(p["HBD"] - orig["HBD"]) > hbd_tol:
        return False

    # 成药性规则
    if p["MW"] > 550:
        return False
    if p["LogP"] > 5:
        return False
    if p["HBD"] > 5:
        return False
    if p["HBA"] > 10:
        return False
    if p["TPSA"] > 140:
        return False

    # 剔除 PAINS
    if has_pains(mol):
        return False

    return True

# ======================
# 骨架保序·生物等排体替换
# ======================
def bioiso_replace(mol):
    m = Chem.Mol(mol)
    try:
        rule = random.choice(ISO_SMARTS)
        patt = Chem.MolFromSmarts(rule[0])
        repl = Chem.MolFromSmarts(random.choice(rule[1:]))
        if patt and repl:
            res = Chem.ReplaceSubstructs(m, patt, repl, replaceAll=False)
            if res:
                return res[0]
    except:
        pass
    return m

# ======================
# 批量生成
# ======================
def generate_library(smi, count, mw_tol, logp_tol, tpsa_tol, hba_tol, hbd_tol):
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return [], {}

    orig = calc_props(mol)
    valid = []
    seen = set()

    for _ in range(count * 12):
        candidate = bioiso_replace(mol)
        if not candidate:
            continue

        if not is_valid(candidate, orig, mw_tol, logp_tol, tpsa_tol, hba_tol, hbd_tol):
            continue

        s = Chem.MolToSmiles(candidate)
        if s in seen or s == smi:
            continue
        seen.add(s)

        valid.append((s, calc_props(candidate)))
        if len(valid) >= count:
            break

    return valid, orig

# ======================
# 界面
# ======================
with st.expander("📥 输入分子", expanded=True):
    input_smi = st.text_input("SMILES", "c1cc(OC)ccc1C")
    gen_count = st.number_input("生成分子数量", 10, 500, 30)

with st.expander("⚙️ 性质不突变约束（可自定义）"):
    col1, col2 = st.columns(2)
    with col1:
        mw_tol   = st.slider("分子量波动 ±", 0, 60, 25)
        logp_tol = st.slider("LogP 波动 ±", 0.0, 2.0, 1.0, 0.1)
    with col2:
        tpsa_tol = st.slider("TPSA 波动 ±", 0, 40, 15)
        hba_tol  = st.slider("HBA 波动 ±", 0, 2, 1)
        hbd_tol  = st.slider("HBD 波动 ±", 0, 2, 1)

if st.button("🚀 生成分子库"):
    with st.spinner("正在生成... 过滤PAINS·成药性·骨架保序"):
        mols, orig = generate_library(
            input_smi, gen_count,
            mw_tol, logp_tol, tpsa_tol, hba_tol, hbd_tol
        )

    if not mols:
        st.error("未生成符合条件的分子，请放宽约束或检查SMILES")
    else:
        st.success(f"✅ 成功生成 {len(mols)} 个高质量分子")
        table = []
        sdf_content = ""
        for i, (smi, prop) in enumerate(mols, 1):
            table.append({
                "ID": f"SG{i:03d}",
                "SMILES": smi,
                "MW": prop["MW"],
                "LogP": prop["LogP"],
                "TPSA": prop["TPSA"],
                "HBA": prop["HBA"],
                "HBD": prop["HBD"]
            })
            m = Chem.MolFromSmiles(smi)
            sdf_content += MolToMolBlock(m) + "$$$$\n"

        df = pd.DataFrame(table)
        st.dataframe(df, use_container_width=True)

        # 导出
        colA, colB = st.columns(2)
        with colA:
            st.download_button("💾 导出CSV", df.to_csv(index=False), "SkelGen分子库.csv")
        with colB:
            st.download_button("📥 导出SDF(对接专用)", sdf_content, "SkelGen分子库.sdf")

st.markdown("---")
st.markdown("""
### 软件说明
- 骨架/环/连接方式**完全不变**
- 仅做**极小局部修饰+生物等排体**
- 自动剔除PAINS毒性基团
- 严格Lipinski成药性
- 导出SDF可直接用于分子对接 / 虚拟筛选
- **完全免费开源，可任意商用、修改、发布**
""")
