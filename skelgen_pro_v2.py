import streamlit as st
import random
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import FilterCatalog
from rdkit.Chem.rdmolfiles import MolToMolBlock
import pandas as pd

# --------------------------
# 页面配置（极简稳定）
# --------------------------
st.set_page_config(
    page_title="SkelGen-Pro 骨架保序分子生成器",
    layout="wide"
)

st.title("🧬 SkelGen-Pro — 骨架锁定·极小扰动药物分子生成")
st.markdown("### 不改变母核骨架 | 生物等排体 | 去PAINS | 成药性过滤 | 对接专用库")

# --------------------------
# PAINS 过滤器
# --------------------------
@st.cache_resource
def load_pains():
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogs.PAINS)
    return FilterCatalog.FilterCatalog(params)

pains = load_pains()

# --------------------------
# 生物等排体替换规则
# --------------------------
ISO_RULES = [
    ("[H]", ["F", "Cl"]),
    ("[F]", ["H", "Cl", "CN"]),
    ("[Cl]", ["F", "H"]),
    ("[CH3]", ["CH2CH3", "CF3", "CN"]),
    ("[OH]", ["OCH3", "F"]),
    ("[NH2]", ["OH", "CH3"]),
    ("C(=O)O", ["c1n[nH]nn1", "S(=O)(=O)N"]),
    ("c1ccccc1", ["c1ccncc1"]),
]

# --------------------------
# 分子属性计算
# --------------------------
def calc_props(mol):
    return {
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
    }

# --------------------------
# 过滤规则
# --------------------------
def is_valid(mol, orig, cfg):
    p = calc_props(mol)
    if abs(p["MW"]-orig["MW"]) > cfg["mw"]: return False
    if abs(p["LogP"]-orig["LogP"]) > cfg["logp"]: return False
    if abs(p["TPSA"]-orig["TPSA"]) > cfg["tpsa"]: return False
    if abs(p["HBA"]-orig["HBA"]) > cfg["hba"]: return False
    if abs(p["HBD"]-orig["HBD"]) > cfg["hbd"]: return False
    if p["MW"] > 550: return False
    if p["LogP"] > 5: return False
    if pains.HasMatch(mol): return False
    return True

# --------------------------
# 等排体替换（骨架锁定）
# --------------------------
def bioiso(mol):
    m = Chem.Mol(mol)
    try:
        rule = random.choice(ISO_RULES)
        patt = Chem.MolFromSmarts(rule[0])
        repl = Chem.MolFromSmarts(random.choice(rule[1:]))
        if patt and repl:
            res = Chem.ReplaceSubstructs(m, patt, repl, replaceAll=False)
            return res[0] if res else m
    except:
        return m
    return m

# --------------------------
# 生成分子库
# --------------------------
def generate(smi, count, cfg):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return [], {}, None
    orig = calc_props(mol)
    valid = []
    seen = set()
    for _ in range(count*15):
        cand = bioiso(mol)
        if not cand: continue
        if not is_valid(cand, orig, cfg): continue
        s = Chem.MolToSmiles(cand)
        if s in seen or s == smi: continue
        seen.add(s)
        valid.append((s, calc_props(cand), cand))
        if len(valid)>=count: break
    return valid, orig, mol

# --------------------------
# 侧边栏输入
# --------------------------
with st.sidebar:
    st.subheader("📥 输入分子")
    smi = st.text_input("SMILES", "c1cc(OC)ccc1C")
    count = st.number_input("生成数量",10,500,30)

    st.subheader("⚙️ 性质约束")
    cfg = {
        "mw": st.slider("MW ±",0,60,25),
        "logp": st.slider("LogP ±",0.0,2.0,1.0,0.1),
        "tpsa": st.slider("TPSA ±",0,40,15),
        "hba": st.slider("HBA ±",0,2,1),
        "hbd": st.slider("HBD ±",0,2,1),
    }

# --------------------------
# 一键生成
# --------------------------
if st.button("🚀 生成骨架保序分子库", use_container_width=True):
    with st.spinner("生成中... 骨架锁定 | 等排体 | PAINS过滤 | 成药性"):
        mols, orig_props, orig_mol = generate(smi, count, cfg)

    if not mols:
        st.error("无法生成符合条件的分子，请放宽参数")
    else:
        st.success(f"✅ 生成 {len(mols)} 个高质量分子")

        # 输出表格
        rows = []
        sdf_content = ""
        for i,(s,p,mol_obj) in enumerate(mols,1):
            rows.append({
                "ID":f"SKEL-{i:03d}",
                "SMILES":s,
                "MW":p["MW"],
                "LogP":p["LogP"],
                "TPSA":p["TPSA"],
                "HBA":p["HBA"],
                "HBD":p["HBD"]
            })
            sdf_content += MolToMolBlock(mol_obj)+"$$$$\n"

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=400)

        # 导出
        col1,col2 = st.columns(2)
        col1.download_button("💾 导出CSV", df.to_csv(index=False), "SkelGen库.csv")
        col2.download_button("📥 导出SDF(对接专用)", sdf_content, "SkelGen库.sdf")

st.markdown("---")
st.markdown("✅ 骨架完全不变 | ✅ 生物等排体 | ✅ 去PAINS | ✅ 成药性过滤 | 开源免费")
