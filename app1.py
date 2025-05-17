import streamlit as st
import pandas as pd
import bcrypt
import os
from PIL import Image
import io
import base64
import joblib
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer
from deap import base, creator, tools, algorithms
import random

# --------------------- 初始化函数 ---------------------
def image_to_base64(image_path):
    """将图片转换为Base64编码"""
    img = Image.open(image_path)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# --------------------- 全局配置 ---------------------
icon_base64 = image_to_base64("图片1.jpg")
background_base64 = image_to_base64("图片1.png")

st.set_page_config(
    page_title="阻燃聚合物复合材料智能设计平台",
    layout="wide",
    page_icon=f"data:image/png;base64,{icon_base64}"
)

# --------------------- 用户认证模块 ---------------------
USERS_FILE = "users.csv"
if not os.path.exists(USERS_FILE):
    pd.DataFrame(columns=["username", "password_hash", "email"]).to_csv(USERS_FILE, index=False)

def load_users():
    return pd.read_csv(USERS_FILE)

def save_user(username, password, email):
    users = load_users()
    if username in users['username'].values:
        return False
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    new_user = pd.DataFrame([[username, password_hash.decode(), email]],
                          columns=["username", "password_hash", "email"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USERS_FILE, index=False)
    return True

def verify_user(username, password):
    users = load_users()
    user = users[users['username'] == username]
    if user.empty:
        return False
    return bcrypt.checkpw(password.encode(), user.iloc[0]['password_hash'].encode())

def reset_password_by_email(email, new_password):
    users = load_users()
    user = users[users['email'] == email]
    if not user.empty:
        password_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
        users.loc[users['email'] == email, 'password_hash'] = password_hash
        users.to_csv(USERS_FILE, index=False)
        return True
    return False

# --------------------- 全局状态 ---------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None

# --------------------- 样式配置 ---------------------
def apply_global_styles():
    """应用全局样式"""
    st.markdown(f"""
    <style>
        :root {{
            --text-base: 1.5rem;   /* 增加基础字号 */
            --text-lg: 1.7rem;     /* 增加较大字号 */
            --text-xl: 1.9rem;     /* 增加超大字号 */
            --title-sm: 2.2rem;    /* 调整标题字号 */
            --title-md: 2.5rem;    /* 调整标题字号 */
            --title-lg: 2.8rem;    /* 调整标题字号 */
            --primary: #1e3d59;
            --secondary: #3f87a6;
            --accent: #2c2c2c;
            --shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}

        /* 统一字体设置 */
        body {{
            font-size: var(--text-base) !important;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }}

        /* 全局头部 */
        .global-header {{
            background: rgba(255,255,255,0.98);
            padding: 2rem 5%;
            box-shadow: var(--shadow);
            margin-bottom: 2rem;
        }}

        .header-title {{
            font-size: var(--title-lg) !important;
            color: var(--primary) !important;
            margin: 0;
            line-height: 1.2;
            font-weight: 600;
        }}

        /* 统一组件样式 */
        .stNumberInput, .stTextInput, .stSelectbox {{
            font-size: var(--text-lg) !important;
        }}

        h1, h2, h3 {{
            color: var(--primary) !important;
        }}

        /* 全局背景图 */
        body {{
            background-image: url("images/图片1.jpg"); /* 修改为你的图片路径 */
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            opacity: 0.9;
        }}

        /* 侧边栏样式 */
        .sidebar {{
            background: rgba(255, 255, 255, 0.9);
        }}
    </style>
    """, unsafe_allow_html=True)




def apply_custom_styles():
    st.markdown(f"""
    <style>
        :root {{
            --text-base: 1.3rem;
            --text-lg: 1.5rem;
            --text-xl: 1.7rem;
            --title-sm: 2.0rem;
            --title-md: 2.3rem;
            --title-lg: 2.6rem;
            --primary: #1e3d59;
            --secondary: #3f87a6;
            --accent: #2c2c2c;
            --shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}

        /* 全局头部 */
        .global-header {{
            background: rgba(255,255,255,0.98);
            padding: 2rem 5%;
            box-shadow: var(--shadow);
            margin-bottom: 3rem;
        }}

        .header-container {{
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            gap: 2rem;
        }}

        .header-logo {{
            width: 120px;
            height: auto;
            border-radius: 12px;
            box-shadow: var(--shadow);
        }}

        .header-title {{
            font-size: 2.8rem !important;
            color: var(--primary) !important;
            margin: 0;
            line-height: 1.2;
        }}

        /* 主内容布局 */
        .main-container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 5%;
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 4rem;
        }}

        /* 核心内容区 */
        .content-section {{
            background: rgba(255,255,255,0.95);
            padding: 3rem;
            border-radius: 20px;
            box-shadow: var(--shadow);
        }}

        /* 登录侧边栏 */
        .auth-sidebar {{
            position: sticky;
            top: 2rem;
            background: rgba(255,255,255,0.98);
            padding: 3rem;
            border-radius: 20px;
            box-shadow: var(--shadow);
            height: fit-content;
        }}

        /* 响应式布局 */
        @media (max-width: 1200px) {{
            .main-container {{
                grid-template-columns: 1fr;
                gap: 2rem;
            }}
            .auth-sidebar {{
                order: -1;
                position: static;
                max-width: 600px;
                margin: 0 auto;
            }}
        }}

        /* 登录表单样式 */
        .auth-form input {{
            font-size: 1.6rem !important;
            padding: 1rem 1.2rem !important;
        }}

        .auth-form button {{
            font-size: 1.8rem !important;
            padding: 1.2rem !important;
            width: 100% !important;
        }}

        /* 内容区块样式 */
        .feature-card {{
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            border-left: 4px solid var(--secondary);
        }}

        .section-title {{
            font-size: var(--title-md);
            color: var(--primary);
            border-bottom: 3px solid var(--secondary);
            padding-bottom: 0.5rem;
            margin-bottom: 2rem;
        }}

        /* 页面背景图设置为base64图片，透明度为99.9% */
        body::before {{
            content: "";
            position: fixed;
            top: 0; left: 0;
            width: 100vw;
            height: 100vh;
            background-image: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA..."); /* 这里替换成实际的base64图片 */
            background-size: cover;
            background-position: center;
            opacity: 0.001;
            z-index: -1;
        }}
    </style>
    """, unsafe_allow_html=True)



# --------------------- 首页内容 ---------------------
def show_homepage():
    apply_custom_styles()

    # 全局头部
    st.markdown(f"""
    <div class="global-header">
        <div class="header-container">
            <img src="data:image/png;base64,{icon_base64}" 
                 class="header-logo"
                 alt="平台标志">
            <h1 class="header-title">阻燃聚合物复合材料智能设计平台</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 主内容容器
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # 左侧核心内容
    with st.container():
        st.markdown('<div class="content-section">', unsafe_allow_html=True)

        # 平台简介
        st.markdown("""
        <div style="font-size:1.5rem; line-height:1.8; margin-bottom:3rem;">
            🚀 本平台融合AI与材料科学技术，致力于高分子复合材料的智能化设计，
            重点关注阻燃性能、力学性能和热稳定性的多目标优化与调控。
        </div>
        """, unsafe_allow_html=True)

        # 核心功能
        st.markdown("""
        <h2 class="section-title">🌟 核心功能</h2>
        <div class="feature-card">
            <h3 style="font-size:1.8rem; color:var(--primary); margin:0 0 1rem 0;">
                🔥 智能性能预测
            </h3>
            <p style="font-size:1.5rem;">
                • 支持LOI（极限氧指数）预测<br>
                • TS（拉伸强度）预测<br>
                
         
        </div>

        <div class="feature-card">
            <h3 style="font-size:1.8rem; color:var(--primary); margin:0 0 1rem 0;">
                ⚗️ 配方优化系统
            </h3>
            <p style="font-size:1.5rem;">
                • 根据输入目标推荐配方<br>
                • 支持选择配方种类<br>
                • 添加剂比例智能推荐
            </p>
        </div>


        """, unsafe_allow_html=True)

        # 研究成果
        st.markdown("""
        <h2 class="section-title">🏆 研究成果</h2>
        <div class="feature-card">
            <p style="font-size:1.5rem;">
                Ma Weibin, Li Ling, Zhang Yu, et al.<br>
                <em>Active learning-based generative design of halogen-free flame-retardant polymeric composites.</em><br>
                <strong>Journal of Materials Informatics</strong> 2025;5:09.<br>
                DOI: <a href="https://doi.org/10.20517/jmi.2025.09" target="_blank" 
                     style="color:var(--secondary); text-decoration:underline;">
                    10.20517/jmi.2025.09
                </a>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # 开发者信息
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            <div class="feature-card">
                <h2 class="section-title">👨💻 开发团队</h2>
                <p style="font-size:1.5rem;">
                    上海大学功能高分子<br>
                    PolyDesign <br>
                    马维宾 | 李凌 | 张瑜<br>
                    宋娜 | 丁鹏
                </p>
            </div>
            """, unsafe_allow_html=True)

        with cols[1]:
            st.markdown("""
            <div class="feature-card">
                <h2 class="section-title">🙏 项目支持</h2>
                <p style="font-size:1.5rem;">
                    云南省科技重点计划<br>
                    项目编号：202302AB080022<br>
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # 结束content-section

    # 右侧登录侧边栏
    with st.container():
        st.markdown('<div class="auth-sidebar">', unsafe_allow_html=True)

        tab_login, tab_register, tab_forgot = st.tabs(["🔐 登录", "📝 注册", "🔑 忘记密码"])

        with tab_login:
            with st.form("login_form", clear_on_submit=True):
                st.markdown('<h2 style="font-size:2rem; text-align:center; margin-bottom:2rem;">用户登录</h2>', 
                          unsafe_allow_html=True)
                username = st.text_input("用户名", key="login_user")
                password = st.text_input("密码", type="password", key="login_pwd")
                
                if st.form_submit_button("立即登录", use_container_width=True):
                    if not all([username, password]):
                        st.error("请输入用户名和密码")
                    elif verify_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.user = username
                        st.rerun()
                    else:
                        st.error("用户名或密码错误")

        with tab_register:
            with st.form("register_form", clear_on_submit=True):
                st.markdown('<h2 style="font-size:2rem; text-align:center; margin-bottom:2rem;">新用户注册</h2>', 
                          unsafe_allow_html=True)
                new_user = st.text_input("用户名（4-20位字母数字）", key="reg_user").strip()
                new_pwd = st.text_input("设置密码（至少6位）", type="password", key="reg_pwd")
                confirm_pwd = st.text_input("确认密码", type="password", key="reg_pwd_confirm")
                email = st.text_input("电子邮箱", key="reg_email")
                
                if st.form_submit_button("立即注册", use_container_width=True):
                    if new_pwd != confirm_pwd:
                        st.error("两次密码输入不一致")
                    elif len(new_user) < 4 or not new_user.isalnum():
                        st.error("用户名格式不正确")
                    elif len(new_pwd) < 6:
                        st.error("密码长度至少6个字符")
                    elif "@" not in email:
                        st.error("请输入有效邮箱地址")
                    else:
                        if save_user(new_user, new_pwd, email):
                            st.success("注册成功！请登录")
                        else:
                            st.error("用户名已存在")

        with tab_forgot:
            with st.form("forgot_form", clear_on_submit=True):
                st.markdown('<h2 style="font-size:2rem; text-align:center; margin-bottom:2rem;">密码重置</h2>', 
                          unsafe_allow_html=True)
                email = st.text_input("注册邮箱", key="reset_email")
                new_password = st.text_input("新密码", type="password", key="new_pwd")
                confirm_password = st.text_input("确认密码", type="password", key="confirm_pwd")
                
                if st.form_submit_button("重置密码", use_container_width=True):
                    if not all([email, new_password, confirm_password]):
                        st.error("请填写所有字段")
                    elif new_password != confirm_password:
                        st.error("两次输入密码不一致")
                    elif reset_password_by_email(email, new_password):
                        st.success("密码已重置，请使用新密码登录")
                    else:
                        st.error("该邮箱未注册")

        st.markdown('</div>', unsafe_allow_html=True)  # 结束auth-sidebar

    st.markdown('</div>', unsafe_allow_html=True)  # 结束main-container

# --------------------- 主流程控制 ---------------------
if not st.session_state.logged_in:
    show_homepage()
    st.stop()



# --------------------- 预测界面 ---------------------
if st.session_state.logged_in:
    # 这里可以放你的后续预测功能代码，例如数据输入、模型预测等




    class Predictor:
        def __init__(self, scaler_path, svc_path):
            self.scaler = joblib.load(scaler_path)
            self.model = joblib.load(svc_path)
            
            # 特征列配置
            self.static_cols = ["产品质量指标_Sn%", "添加比例", "一甲%"]
            self.time_series_cols = [
                "黄度值_3min", "6min", "9min", "12min",
                "15min", "18min", "21min", "24min"
            ]
            self.eng_features = [
                'seq_length', 'max_value', 'mean_value', 'min_value',
                'std_value', 'trend', 'range_value', 'autocorr'
            ]
            self.imputer = SimpleImputer(strategy="mean")
    
        def _truncate(self, df):
            time_cols = [col for col in df.columns if "min" in col.lower()]
            time_cols_ordered = [col for col in df.columns if col in time_cols]
            if time_cols_ordered:
                row = df.iloc[0][time_cols_ordered]
                if row.notna().any():
                    max_idx = row.idxmax()
                    max_pos = time_cols_ordered.index(max_idx)
                    for col in time_cols_ordered[max_pos + 1:]:
                        df.at[df.index[0], col] = np.nan
            return df
        
        def _get_slope(self, row, col=None):
            # col 是可选的，将被忽略
            x = np.arange(len(row))
            y = row.values
            mask = ~np.isnan(y)
            if sum(mask) >= 2:
                return stats.linregress(x[mask], y[mask])[0]
            return np.nan
    
        def _calc_autocorr(self, row):
            """计算一阶自相关系数"""
            values = row.dropna().values
            if len(values) > 1:
                n = len(values)
                mean = np.mean(values)
                numerator = sum((values[:-1] - mean) * (values[1:] - mean))
                denominator = sum((values - mean) ** 2)
                if denominator != 0:
                    return numerator / denominator
            return np.nan
    
        def _extract_time_series_features(self, df):
            """修复后的时序特征提取"""
            time_data = df[self.time_series_cols]
            time_data_filled = time_data.ffill(axis=1)
            
            features = pd.DataFrame()
            features['seq_length'] = time_data_filled.notna().sum(axis=1)
            features['max_value'] = time_data_filled.max(axis=1)
            features['mean_value'] = time_data_filled.mean(axis=1)
            features['min_value'] = time_data_filled.min(axis=1)
            features['std_value'] = time_data_filled.std(axis=1)
            features['range_value'] = features['max_value'] - features['min_value']
            features['trend'] = time_data_filled.apply(self._get_slope, axis=1)
            features['autocorr'] = time_data_filled.apply(self._calc_autocorr, axis=1)
            return features
    
        def predict_one(self, sample):
            full_cols = self.static_cols + self.time_series_cols
            df = pd.DataFrame([sample], columns=full_cols)
            df = self._truncate(df)
            
            # 特征合并
            static_features = df[self.static_cols]
            time_features = self._extract_time_series_features(df)
            feature_df = pd.concat([static_features, time_features], axis=1)
            feature_df = feature_df[self.static_cols + self.eng_features]
            
            # 验证维度
            if feature_df.shape[1] != self.scaler.n_features_in_:
                raise ValueError(f"特征维度不匹配！当前：{feature_df.shape[1]}，需要：{self.scaler.n_features_in_}")
            
            X_scaled = self.scaler.transform(feature_df)
            return self.model.predict(X_scaled)[0]

    # 侧边栏主导航
    page = st.sidebar.selectbox(
        "🔧 主功能选择",
        ["性能预测", "配方建议"],
        key="main_nav"
    )

    # 子功能选择（仅在配方建议时显示）
    sub_page = None
    if page == "配方建议":
        sub_page = st.sidebar.selectbox(
            "🔧 子功能选择",
            ["配方优化", "添加剂推荐"],
            key="sub_nav"
        )
    with st.sidebar:
        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state.logged_in = False  # 设置登录状态为 False
            st.session_state.user = None  # 清除用户信息
            st.success("已成功退出登录")  # 显示成功消息
            st.rerun()  # 重新加载页面

    @st.cache_resource  # 更新后的缓存装饰器
    def load_models():
        # 确保模型文件路径正确
        loi_data = joblib.load("model_and_scaler_loi.pkl")
        ts_data = joblib.load("model_and_scaler_ts1.pkl")
        return {
            "loi_model": loi_data["model"],
            "loi_scaler": loi_data["scaler"],
            "ts_model": ts_data["model"],
            "ts_scaler": ts_data["scaler"],
            "loi_features": pd.read_excel("trainrg3.xlsx").drop(columns="LOI", errors='ignore').columns.tolist(),
            "ts_features": pd.read_excel("trainrg3TS.xlsx").drop(columns="TS", errors='ignore').columns.tolist(),
        }
    
    models = load_models()
        
    # 获取单位
    def get_unit(fraction_type):
        if fraction_type == "质量":
            return "g"
        elif fraction_type == "质量分数":
            return "wt%"
        elif fraction_type == "体积分数":
            return "vol%"
    
    # 保证PP在首列
    def ensure_pp_first(features):
        if "PP" in features:
            features.remove("PP")
        return ["PP"] + sorted(features)
      

    if page == "性能预测":
        apply_global_styles()
        render_global_header()
        st.subheader("🔮 性能预测：基于配方预测LOI和TS")
    
        # 初始化 input_values
        if 'input_values' not in st.session_state:
            st.session_state.input_values = {}  # 使用会话状态保存输入值
    
        # 基体材料数据
        matrix_materials = {
            "PP": {"name": "Polypropylene", "full_name": "Polypropylene (PP)", "range": (53.5, 99.5)},
            "PA": {"name": "Polyamide", "full_name": "Polyamide (PA)", "range": (0, 100)},
            "PC/ABS": {"name": "Polycarbonate/Acrylonitrile Butadiene Styrene Blend", "full_name": "Polycarbonate/Acrylonitrile Butadiene Styrene Blend (PC/ABS)", "range": (0, 100)},
            "POM": {"name": "Polyoxymethylene", "full_name": "Polyoxymethylene (POM)", "range": (0, 100)},
            "PBT": {"name": "Polybutylene Terephthalate", "full_name": "Polybutylene Terephthalate (PBT)", "range": (0, 100)},
            "PVC": {"name": "Polyvinyl Chloride", "full_name": "Polyvinyl Chloride (PVC)", "range": (0, 100)},
        }
    
        # 阻燃剂数据
        flame_retardants = {
            "AHP": {"name": "Aluminum Hyphosphite", "range": (0, 25)},
            "CFA": {"name": "Carbon Forming agent", "range": (0, 10)},
            "ammonium octamolybdate": {"name": "Ammonium Octamolybdate", "range": (0, 3.4)},
            "Al(OH)3": {"name": "Aluminum Hydroxide", "range": (0, 10)},
            "APP": {"name": "Ammonium Polyphosphate", "range": (0, 19.5)},
            "Pentaerythritol": {"name": "Pentaerythritol", "range": (0, 1.3)},
            "DOPO": {"name": "9,10-Dihydro-9-oxa-10-phosphaphenanthrene-10-oxide", "range": (0, 27)},
            "XS-FR-8310": {"name": "XS-FR-8310", "range": (0, 35)},
            "ZS": {"name": "Zinc Stannate", "range": (0, 34.5)},
            "XiuCheng": {"name": "XiuCheng Flame Retardant", "range": (0, 35)},
            "ZHS": {"name": "Hydroxy Zinc Stannate", "range": (0, 34.5)},
            "ZnB": {"name": "Zinc Borate", "range": (0, 2)},
            "antimony oxides": {"name": "Antimony Oxides", "range": (0, 2)},
            "Mg(OH)2": {"name": "Magnesium Hydroxide", "range": (0, 34.5)},
            "TCA": {"name": "Triazine Carbonization Agent", "range": (0, 17.4)},
            "MPP": {"name": "Melamine Polyphosphate", "range": (0, 25)},
            "PAPP": {"name": "Piperazine Pyrophosphate", "range": (0, 24.5)},
            "其他": {"name": "Other", "range": (0, 100)},
        }
    
        # 助剂数据
        additives = {
            "processing additives": {
                "Anti-drip-agent": {"name": "Polytetrafluoroethylene Anti-dripping Agent", "range": (0, 0.3)},
                "ZBS-PV-OA": {"name": "Zinc Borate Stabilizer PV-OA Series", "range": (0, 35)},
                "FP-250S": {"name": "Processing Aid FP-250S (Acrylic)", "range": (0, 35)},
            },
            "Fillers": {
                "wollastonite": {"name": "Wollastonite (Calcium Metasilicate)", "range": (0, 5)},
                "SiO2": {"name": "Silicon Dioxide", "range": (0, 6)},
            },
            "Coupling Agents": {
                "silane coupling agent": {"name": "Amino Silane Coupling Agent", "range": (0.5, 3)},
            },
            "Antioxidants": {
                "antioxidant": {"name": "Irganox 1010 Antioxidant", "range": (0.1, 0.5)},
            },
            "Lubricants": {
                "M-2200B": {"name": "Lubricant M-2200B (Ester-based)", "range": (0.5, 3)},
            },
            "Functional Additives": {  # 替换Others为功能助剂
                "Custom Additive": {"name": "Custom Additive", "range": (0, 5)},
            },
        }
    
        fraction_type = st.sidebar.selectbox("选择输入的单位", ["质量", "质量分数", "体积分数"])
    
        # 配方成分部分（基体和阻燃剂）
        st.subheader("请选择配方成分")
        col_matrix = st.columns([4, 3], gap="medium")  # 调整列宽比例
        with col_matrix[0]:
            selected_matrix = st.selectbox("选择基体材料", [matrix_materials[key]["full_name"] for key in matrix_materials], index=0)
            # 获取选中基体的缩写
            matrix_key = [key for key in matrix_materials if matrix_materials[key]["full_name"] == selected_matrix][0]
            matrix_name = matrix_materials[matrix_key]["name"]
            matrix_range = matrix_materials[matrix_key]["range"]
            # 显示推荐范围，不带单位
            st.markdown(f"**推荐范围**: {matrix_range[0]} - {matrix_range[1]}")
    
        with col_matrix[1]:
            unit_matrix = "g" if fraction_type == "质量" else ("%" if fraction_type == "质量分数" else "vol%")
            st.session_state.input_values[matrix_key] = st.number_input(
                f"{matrix_name} 含量 ({unit_matrix})", min_value=0.0, max_value=100.0, value=50.0, step=0.1
            )
    
        # ========== 阻燃剂显示 ==========  
        st.subheader("请选择阻燃剂")
        
        # 显示完整名称的下拉框
        selected_flame_retardants = st.multiselect(
            "选择阻燃剂（必选锡酸锌和羟基锡酸锌）", 
            [flame_retardants[key]["name"] for key in flame_retardants],
            default=[flame_retardants[list(flame_retardants.keys())[0]]["name"]]
        )
        
        # 根据选择的完整名称，设置输入框
        for flame_name in selected_flame_retardants:
            # 获取对应的阻燃剂缩写
            for key, value in flame_retardants.items():
                if value["name"] == flame_name:
                    flame_info = value
                    with st.expander(f"{flame_info['name']} 推荐范围"):
                        st.write(f"推荐范围：{flame_info['range'][0]} - {flame_info['range'][1]}")  # 不带单位
                        unit_add = "g" if fraction_type == "质量" else ("%" if fraction_type == "质量分数" else "vol%")
                        
                        # 设置默认值，确保它不小于最小值
                        min_val = float(flame_info['range'][0])
                        max_val = float(flame_info['range'][1])
                        default_value = max(min_val, 0.0)
    
                        # 使用 number_input 输入框
                        st.session_state.input_values[key] = st.number_input(
                            f"{flame_info['name']} 含量 ({unit_add})", 
                            min_value=min_val, 
                            max_value=max_val, 
                            value=default_value, 
                            step=0.1,
                            key=f"fr_{key}"
                        )
    
        # ========== 助剂显示 ==========  
        st.subheader("选择助剂")
        selected_additives = st.multiselect(
            "选择助剂（可多选）", list(additives.keys()), default=[list(additives.keys())[0]]
        )
        
        for category in selected_additives:
            for ad, additive_info in additives[category].items():
                with st.expander(f"{additive_info['name']} 推荐范围"):
                    st.write(f"推荐范围：{additive_info['range'][0]} - {additive_info['range'][1]}")  # 不带单位
                    unit_additive = "g" if fraction_type == "质量" else ("%" if fraction_type == "质量分数" else "vol%")
                    min_additive = float(additive_info["range"][0])
                    max_additive = float(additive_info["range"][1])
                    default_additive = max(min_additive, 0.0)
    
                    # 设置助剂输入框
                    st.session_state.input_values[ad] = st.number_input(
                        f"{additive_info['name']} 含量 ({unit_additive})", 
                        min_value=min_additive, 
                        max_value=max_additive, 
                        value=default_additive, 
                        step=0.1,
                        key=f"additive_{ad}"
                    )
            
            # 校验和预测
            total = sum(st.session_state.input_values.values())  # 总和计算
            is_only_pp = all(v == 0 for k, v in st.session_state.input_values.items() if k != "PP")  # 仅PP配方检查
        
        with st.expander("✅ 输入验证"):
            if fraction_type in ["体积分数", "质量分数"]:
                if abs(total - 100.0) > 1e-6:
                    st.error(f"❗ {fraction_type}的总和必须为100%（当前：{total:.2f}%）")
                else:
                    st.success(f"{fraction_type}总和验证通过")
            else:
                st.success("成分总和验证通过")
                if is_only_pp:
                    st.info("检测到纯PP配方")
        
            # 验证配方是否包含锡酸锌或羟基锡酸锌
            selected_flame_keys = [key for key in flame_retardants if flame_retardants[key]["name"] in selected_flame_retardants]
            if not any("Zinc Stannate" in flame_retardants[key]["name"] or "Hydroxy Zinc Stannate" in flame_retardants[key]["name"] for key in selected_flame_keys):
                st.error("❗ 配方必须包含锡酸锌（Zinc Stannate）或羟基锡酸锌（Hydroxy Zinc Stannate）。")
            else:
                st.success("配方验证通过，包含锡酸锌或羟基锡酸锌。")
            
            # 验证并点击“开始预测”按钮
            if st.button("🚀 开始预测", type="primary"):
                # 检查输入总和是否为100%，如果不是则停止
                if fraction_type in ["体积分数", "质量分数"] and abs(total - 100.0) > 1e-6:
                    st.error(f"预测中止：{fraction_type}的总和必须为100%")
                    st.stop()
        
                # 如果是纯PP配方，直接给出模拟值
                if is_only_pp:
                    loi_pred = 17.5
                    ts_pred = 35.0
                else:
                    # 体积分数转换为质量分数
                    if fraction_type == "体积分数":
                        vol_values = np.array(list(st.session_state.input_values.values()))
                        total_mass = vol_values.sum()
                        mass_values = vol_values * total_mass  # 按比例转换
                        st.session_state.input_values = {k: (v / total_mass * 100) for k, v in zip(st.session_state.input_values.keys(), mass_values)}
        
                    # 填充缺失的特征值
                    for feature in models["loi_features"]:
                        if feature not in st.session_state.input_values:
                            st.session_state.input_values[feature] = 0.0
        
                    loi_input = np.array([[st.session_state.input_values[f] for f in models["loi_features"]]])
                    loi_scaled = models["loi_scaler"].transform(loi_input)
                    loi_pred = models["loi_model"].predict(loi_scaled)[0]
        
                    # 处理TS预测
                    for feature in models["ts_features"]:
                        if feature not in st.session_state.input_values:
                            st.session_state.input_values[feature] = 0.0
        
                    ts_input = np.array([[st.session_state.input_values[f] for f in models["ts_features"]]])
                    ts_scaled = models["ts_scaler"].transform(ts_input)
                    ts_pred = models["ts_model"].predict(ts_scaled)[0]
        
                # 显示预测结果
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="LOI预测值", value=f"{loi_pred:.2f}%")
                with col2:
                    st.metric(label="TS预测值", value=f"{ts_pred:.2f} MPa")

    



    
    elif page == "配方建议":
        apply_global_styles()
        render_global_header()
        if sub_page == "配方优化":
            fraction_type = st.sidebar.radio(
                "📐 单位类型",
                ["质量", "质量分数", "体积分数"],
                key="unit_type"
            )
            st.subheader("🧪 配方建议：根据目标LOI和TS优化配方")
        
            matrix_materials = ["PP", "PA", "PC/ABS", "POM", "PBT", "PVC", "其他"]
            flame_retardants = [
                    "AHP", "ammonium octamolybdate", "Al(OH)3", "CFA", "APP", "Pentaerythritol", "DOPO",
                    "EPFR-1100NT", "XS-FR-8310", "ZS", "XiuCheng", "ZHS", "ZnB", "antimony oxides",
                    "Mg(OH)2", "TCA", "MPP", "PAPP", "其他"
                ]
            additives = [
                    "Anti-drip-agent", "wollastonite", "M-2200B", "ZBS-PV-OA", "FP-250S", "silane coupling agent", "antioxidant",
                    "SiO2", "其他"
                ]
        
            selected_matrix = st.selectbox("选择基体", matrix_materials, index=0)
            selected_flame_retardants = st.multiselect("选择阻燃剂", flame_retardants, default=["ZS"])
            selected_additives = st.multiselect("选择助剂", additives, default=["wollastonite"])
        
            target_loi = st.number_input("目标LOI值（%）", min_value=0.0, max_value=100.0, value=30.0)
            target_ts = st.number_input("目标TS值（MPa）", min_value=0.0, value=40.0)
        
            if st.button("🚀 开始优化"):
                all_features = [selected_matrix] + selected_flame_retardants + selected_additives
        
                creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
                creator.create("Individual", list, fitness=creator.FitnessMin)
        
                toolbox = base.Toolbox()
        
                def repair_individual(individual):
                    """确保基体含量最大且总和为100%"""
                    individual = [max(0.0, x) for x in individual]
                    total = sum(individual)
                    
                    if total <= 1e-6:
                        return [100.0/len(individual)]*len(individual)
                    
                    scale = 100.0 / total
                    individual = [x*scale for x in individual]
                    
                    try:
                        matrix_idx = all_features.index(selected_matrix)
                        matrix_value = individual[matrix_idx]
                        other_max = max([v for i,v in enumerate(individual) if i != matrix_idx], default=0)
                        
                        if matrix_value <= other_max:
                            delta = other_max - matrix_value + 0.01
                            others_total = sum(v for i,v in enumerate(individual) if i != matrix_idx)
                            
                            if others_total > 0:
                                deduction_ratio = delta / others_total
                                for i in range(len(individual)):
                                    if i != matrix_idx:
                                        individual[i] *= (1 - deduction_ratio)
                                individual[matrix_idx] += delta*others_total/others_total
                            
                            total = sum(individual)
                            scale = 100.0 / total
                            individual = [x*scale for x in individual]
                            
                    except ValueError:
                        pass
                    
                    return individual
        
                def generate_individual():
                    """生成初始个体，确保基体含量占优"""
                    try:
                        matrix_idx = all_features.index(selected_matrix)
                    except ValueError:
                        matrix_idx = 0
        
                    matrix_range = (60, 100) if selected_matrix == "PP" else (30, 50)
                    matrix_percent = random.uniform(*matrix_range)
                    
                    remaining = 100 - matrix_percent
                    n_others = len(all_features) - 1
                    
                    if n_others == 0:
                        return [matrix_percent]
                    
                    others = np.random.dirichlet(np.ones(n_others)*0.5) * remaining
                    others = others.tolist()
                    
                    individual = [0.0]*len(all_features)
                    individual[matrix_idx] = matrix_percent
                    
                    other_idx = 0
                    for i in range(len(all_features)):
                        if i != matrix_idx:
                            individual[i] = others[other_idx]
                            other_idx += 1
                            
                    return repair_individual(individual)
        
                toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
                toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
                def evaluate(individual):
                    try:
                        input_values = dict(zip(all_features, individual))
                        
                        # LOI预测部分
                        loi_input = np.array([[input_values.get(f, 0.0) for f in models["loi_features"]]])
                        loi_scaled = models["loi_scaler"].transform(loi_input)
                        loi_pred = models["loi_model"].predict(loi_scaled)[0]
        
                        # TS预测部分
                        ts_input = np.array([[input_values.get(f, 0.0) for f in models["ts_features"]]])
                        ts_scaled = models["ts_scaler"].transform(ts_input)
                        ts_pred = models["ts_model"].predict(ts_scaled)[0]
        
                        return (abs(target_loi - loi_pred), abs(target_ts - ts_pred))
                    except Exception as e:
                        print(f"Error in evaluate: {e}")
                        return (float('inf'), float('inf'))
        
                def cxBlendWithConstraint(ind1, ind2, alpha):
                    tools.cxBlend(ind1, ind2, alpha)
                    ind1[:] = repair_individual(ind1)
                    ind2[:] = repair_individual(ind2)
                    return ind1, ind2
        
                def mutGaussianWithConstraint(individual, mu, sigma, indpb):
                    tools.mutGaussian(individual, mu, sigma, indpb)
                    individual[:] = repair_individual(individual)
                    return individual,
        
                toolbox.register("evaluate", evaluate)
                toolbox.register("mate", cxBlendWithConstraint, alpha=0.5)
                toolbox.register("mutate", mutGaussianWithConstraint, mu=0, sigma=3, indpb=0.2)
                toolbox.register("select", tools.selNSGA2)
        
                population = toolbox.population(n=150)
                algorithms.eaMuPlusLambda(
                    population, toolbox,
                    mu=150, lambda_=300,
                    cxpb=0.7, mutpb=0.3,
                    ngen=250, verbose=False
                )
        
                # 获取符合条件的个体并计算最终结果
                valid_individuals = [ind for ind in population if not np.isinf(ind.fitness.values[0])]
                best_individuals = tools.selBest(valid_individuals, k=5)
        
                results = []
                for ind in best_individuals:
                    normalized = [round(x, 2) for x in repair_individual(ind)]
                    matrix_value = normalized[all_features.index(selected_matrix)]
                    
                    if not all(v <= matrix_value for i,v in enumerate(normalized) if i != all_features.index(selected_matrix)):
                        continue
                        
                    input_dict = dict(zip(all_features, normalized))
                    
                    # LOI预测部分
                    loi_input = [[input_dict.get(f, 0) for f in models["loi_features"]]]
                    loi_scaled = models["loi_scaler"].transform(loi_input)
                    loi_pred = models["loi_model"].predict(loi_scaled)[0]
                    
                    # TS预测部分
                    ts_input = [[input_dict.get(f, 0) for f in models["ts_features"]]]
                    ts_scaled = models["ts_scaler"].transform(ts_input)
                    ts_pred = models["ts_model"].predict(ts_scaled)[0]
        
                    if abs(target_loi - loi_pred) > 10 or abs(target_ts - ts_pred) > 10:
                        continue
                    
                    results.append({
                        **{f: normalized[i] for i,f in enumerate(all_features)},
                        "LOI预测值 (%)": round(loi_pred, 2),
                        "TS预测值 (MPa)": round(ts_pred, 2),
                    })
        
                if results:
                    df = pd.DataFrame(results)
                    unit = "wt%" if "质量分数" in fraction_type else "vol%" if "体积分数" in fraction_type else "g"
                    df.columns = [f"{col} ({unit})" if col in all_features else col for col in df.columns]
                    
                    st.dataframe(
                        df.style.apply(lambda x: ["background: #e6ffe6" if x["LOI预测值 (%)"] >= target_loi and 
                                                x["TS预测值 (MPa)"] >= target_ts else "" for _ in x], axis=1),
                        height=400
                    )
                else:
                    st.warning("未找到符合要求的配方，请尝试调整目标值")
    
        
        elif sub_page == "添加剂推荐":
            st.subheader("🧪 PVC添加剂智能推荐")
            predictor = Predictor("scaler_fold_1.pkl", "svc_fold_1.pkl")
            with st.expander("点击查看参考样本"):
                st.markdown("""
                ### 参考样本
                以下是一些参考样本，展示了不同的输入数据及对应的推荐添加剂类型：
                """)
                
                    # 参考样本数据
                sample_data = [
                    ["样本1", "无添加剂", 
                     {"Sn%": 19.2, "添加比例": 0, "一甲%": 32, "黄度值_3min": 5.36, "黄度值_6min": 6.29, "黄度值_9min": 7.57, "黄度值_12min": 8.57, "黄度值_15min": 10.26, "黄度值_18min": 13.21, "黄度值_21min": 16.54, "黄度值_24min": 27.47}],
                    ["样本2", "氯化石蜡", 
                     {"Sn%": 18.5, "添加比例": 3.64, "一甲%": 31.05, "黄度值_3min": 5.29, "黄度值_6min": 6.83, "黄度值_9min": 8.00, "黄度值_12min": 9.32, "黄度值_15min": 11.40, "黄度值_18min": 14.12, "黄度值_21min": 18.37, "黄度值_24min": 30.29}],
                    ["样本3", "EA15（市售液体钙锌稳定剂）", 
                     {"Sn%": 19, "添加比例": 1.041666667, "一甲%": 31.88, "黄度值_3min": 5.24, "黄度值_6min": 6.17, "黄度值_9min": 7.11, "黄度值_12min": 8.95, "黄度值_15min": 10.33, "黄度值_18min": 13.21, "黄度值_21min": 17.48, "黄度值_24min": 28.08}]
                ]
    
                # 为每个样本创建一个独立的表格
                for sample in sample_data:
                    sample_name, additive, features = sample
                    st.markdown(f"#### {sample_name} - {additive}")
                    
                    # 将数据添加到表格
                    features["推荐添加剂"] = additive  # 显示样本推荐的添加剂
                    features["推荐添加量 (%)"] = features["添加比例"]  # 使用已提供的添加比例
                    
                    # 转换字典为 DataFrame
                    df_sample = pd.DataFrame(list(features.items()), columns=["特征", "值"])
                    st.table(df_sample)  # 显示为表格形式
    # 修改黄度值输入为独立输入
            with st.form("additive_form"):
                st.markdown("### 基础参数")
                col_static = st.columns(3)
                with col_static[0]:
                    add_ratio = st.number_input("添加比例 (%)", 
                                              min_value=0.0,
                                              max_value=100.0,
                                              value=3.64,
                                              step=0.1)
                with col_static[1]:
                    sn_percent = st.number_input("Sn含量 (%)", 
                                               min_value=0.0, 
                                               max_value=100.0,
                                               value=18.5,
                                               step=0.1,
                                               help="锡含量范围0%~100%")
                with col_static[2]:
                    yijia_percent = st.number_input("一甲含量 (%)",
                                                   min_value=0.0,
                                                   max_value=100.0,
                                                   value=31.05,
                                                   step=0.1,
                                                   help="一甲胺含量范围15.1%~32%")
                
                st.markdown("### 黄度值")
                yellow_values = {}
                col1, col2, col3, col4 = st.columns(4)
                yellow_values["3min"] = st.number_input("3min 黄度值", min_value=0.0, max_value=100.0, value=5.29, step=0.1)
                yellow_values["6min"] = st.number_input("6min 黄度值", min_value=yellow_values["3min"], max_value=100.0, value= 6.83, step=0.1)
                yellow_values["9min"] = st.number_input("9min 黄度值", min_value=yellow_values["6min"], max_value=100.0, value=8.00, step=0.1)
                yellow_values["12min"] = st.number_input("12min 黄度值", min_value=yellow_values["9min"], max_value=100.0, value=9.32, step=0.1)
                yellow_values["15min"] = st.number_input("15min 黄度值", min_value=yellow_values["12min"], max_value=100.0, value=11.40, step=0.1)
                yellow_values["18min"] = st.number_input("18min 黄度值", min_value=yellow_values["15min"], max_value=100.0, value=14.12, step=0.1)
                yellow_values["21min"] = st.number_input("21min 黄度值", min_value=yellow_values["18min"], max_value=100.0, value=18.37, step=0.1)
                yellow_values["24min"] = st.number_input("24min 黄度值", min_value=yellow_values["21min"], max_value=100.0, value=30.29, step=0.1)
            
                submit_btn = st.form_submit_button("生成推荐方案")
            
            # 如果提交了表单，进行数据验证和预测
            if submit_btn:
                # 验证比例是否符合要求：每个黄度值输入必须满足递增条件
                if any(yellow_values[t] > yellow_values[next_time] for t, next_time in zip(yellow_values.keys(), list(yellow_values.keys())[1:])):
                    st.error("错误：黄度值必须随时间递增！请检查输入数据")
                    st.stop()
                
                # 构建输入样本
                sample = [
                    sn_percent, add_ratio, yijia_percent,
                    yellow_values["3min"], yellow_values["6min"],
                    yellow_values["9min"], yellow_values["12min"],
                    yellow_values["15min"], yellow_values["18min"],
                    yellow_values["21min"], yellow_values["24min"]
                ]
            
                # 进行预测
                prediction = predictor.predict_one(sample)
                result_map = {
                    1: "无推荐添加剂", 
                    2: "氯化石蜡", 
                    3: "EA12（脂肪酸复合醇酯）",
                    4: "EA15（市售液体钙锌稳定剂）", 
                    5: "EA16（环氧大豆油）",
                    6: "G70L（多官能团的脂肪酸复合酯混合物）", 
                    7: "EA6（亚磷酸酯）"
                }
            
                # 动态确定添加量和显示名称
                additive_amount = 0.0 if prediction == 1 else add_ratio
                additive_name = result_map[prediction]
            
                # 构建配方表
                formula_data = [
                    ["PVC份数", 100.00],
                    ["加工助剂ACR份数", 1.00],
                    ["外滑剂70S份数", 0.35],
                    ["MBS份数", 5.00],
                    ["316A份数", 0.20],
                    ["稳定剂份数", 1.00]
                ]
            
                if prediction != 1:
                    formula_data.append([f"{additive_name}含量（wt%）", additive_amount])
                else:
                    formula_data.append([additive_name, additive_amount])
            
                # 创建格式化表格
                df = pd.DataFrame(formula_data, columns=["材料名称", "含量"])
                styled_df = df.style.format({"含量": "{:.2f}"})\
                                      .hide(axis="index")\
                                      .set_properties(**{'text-align': 'left'})
            
                # 展示推荐结果
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.success(f"**推荐添加剂类型**  \n{additive_name}")
                    st.metric("建议添加量", 
                             f"{additive_amount:.2f}%",
                             delta="无添加" if prediction == 1 else None)
                with col2:
                    st.markdown("**完整配方表（基于PVC 100份）**")
                    st.dataframe(styled_df,
                                 use_container_width=True,
                                 height=280,
                                 column_config={
                                     "材料名称": "材料名称",
                                     "含量": st.column_config.NumberColumn(
                                         "含量",
                                         format="%.2f"
                                     )
                                 })
    
    
    
    
    # 添加页脚
    def add_footer():
        st.markdown("""
        <hr>
        <footer style="text-align: center;">
            <p>© 2025 阻燃聚合物复合材料智能设计平台</p>
            <p>声明：本平台仅供学术研究、技术验证等非营利性科研活动使用，严禁用于任何商业用途。</p>
        </footer>
        """, unsafe_allow_html=True)
    
    add_footer()
