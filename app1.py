import pandas as pd
import numpy as np
import warnings
from scipy import stats
from scipy.stats import kurtosis, skew
from sklearn.impute import SimpleImputer
import joblib
from deap import base, creator, tools, algorithms
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import random

# 页面配置
# 页面配置
import base64
from PIL import Image
import io

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
st.title("账号登录与预测")

# 手机号和验证码输入
phone_number = st.text_input("请输入您的手机号", "")
code = st.text_input("请输入验证码", "")

# 发送验证码按钮
if st.button("发送验证码"):
    if phone_number:
        sent_code = send_verification_code(phone_number)
        st.success(f"验证码已发送：{sent_code}")
    else:
        st.error("请先输入手机号")

# 验证验证码按钮
if st.button("验证验证码"):
    if verify_code(phone_number, code):
        st.success("验证码验证成功！")




        def image_to_base64(image_path, quality=95):
            """高质量图片转base64"""
            img = Image.open(image_path)
            
            # 保持原始分辨率进行缩放
            if img.width != 1000:
                img = img.resize((1000, int(img.height * (1000 / img.width))), 
                                resample=Image.LANCZOS)
            
            buffered = io.BytesIO()
            img.save(buffered, format="PNG", quality=quality, optimize=True)
            return base64.b64encode(buffered.getvalue()).decode()
        
        # 页面配置
        image_path = "图片1.jpg"
        icon_base64 = image_to_base64(image_path)  # 质量参数设为95
        
        st.set_page_config(
            page_title="阻燃聚合物复合材料智能设计平台",
            layout="wide",
            page_icon=f"data:image/png;base64,{icon_base64}"
        )
        
        # 获取精确尺寸
        img = Image.open(image_path)
        target_width = 800
        target_height = int(img.height * (target_width / img.width))
        
        # 图片显示样式
        st.markdown(f"""
        <style>
            .fixed-width-img {{
                width: {target_width}px !important;
                height: {target_height}px !important;
                object-fit: contain;
                margin-left: 0;
                padding: 0;
                image-rendering: -webkit-optimize-contrast; /* Safari */
                image-rendering: crisp-edges; /* Standard */
            }}
            
            @media (max-width: 1050px) {{
                .fixed-width-img {{
                    width: 95% !important;
                    height: auto !important;
                    max-width: 1000px;
                }}
            }}
        </style>
        """, unsafe_allow_html=True)
        
        # 全局页眉样式
        st.markdown("""
        <style>
            .global-header {
                display: flex;
                align-items: center;
                gap: 25px;
                margin: 0 0 2rem 0;
                padding: 1rem 0;
                border-bottom: 3px solid #1e3d59;
                position: sticky;
                top: 0;
                background: white;
                z-index: 1000;
            }
            
            .header-logo {
                width: 80px;
                height: auto;
                flex-shrink: 0;
                border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            }
            
            .header-title {
                font-size: 4.4rem !important;
                color: #1e3d59;
                margin: 0;
                line-height: 1.2;
                font-family: 'Microsoft YaHei', sans-serif;
            }
            
            .header-subtitle {
                font-size: 1.1rem;
                color: #3f87a6;
                margin: 0.3rem 0 0 0;
            }
        
            @media (max-width: 768px) {
                .global-header {
                    gap: 15px;
                    padding: 0.5rem 0;
                }
                
                .header-logo {
                    width: 60px;
                }
                
                .header-title {
                    font-size: 1.8rem !important;
                }
                
                .header-subtitle {
                    font-size: 0.9rem;
                }
            }
        </style>
        """, unsafe_allow_html=True)
        
        # 全局页眉HTML
        st.markdown(f"""
        <div class="global-header">
            <img src="data:image/png;base64,{icon_base64}" 
                 class="header-logo"
                 alt="Platform Logo">
            <div>
                <h1 class="header-title">阻燃聚合物复合材料智能设计平台</h1>
                <p class="header-subtitle">Flame Retardant Polymer Composite Intelligent Platform</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        # 侧边栏主导航
        page = st.sidebar.selectbox(
            "🔧 主功能选择",
            ["首页","性能预测", "配方建议"],
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
        
        # 加载模型
        @st.cache_resource
        def load_models():
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
        
        
        # 首页
        if page == "首页":
            st.markdown("""
            <style>
                :root {
                    /* 字号系统 */
                    --text-base: 1.15rem;
                    --text-lg: 1.3rem;
                    --text-xl: 1.5rem;
                    --title-sm: 1.75rem;
                    --title-md: 2rem;
                    --title-lg: 2.25rem;
                    
                    /* 颜色系统 */
                    --primary: #1e3d59;
                    --secondary: #3f87a6;
                    --accent: #2c2c2c;
                }
        
                body {
                    /* 中文字体优先使用微软雅黑，英文使用Times New Roman */
                    font-family: "Times New Roman", "微软雅黑", SimSun, serif;
                    font-size: var(--text-base);
                    line-height: 1.7;
                    color: var(--accent);
                }
        
                /* 标题系统 */
                .platform-title {
                    font-family: "Times New Roman", "微软雅黑", SimSun, serif;
                    font-size: var(--title-lg);
                    font-weight: 600;
                    color: var(--primary);
                    margin: 0 0 1.2rem 1.5rem;
                    line-height: 1.3;
                }
        
                .section-title {
                    font-family: "Times New Roman", "微软雅黑", SimSun, serif;
                    font-size: var(--title-md);
                    font-weight: 600;
                    color: var(--primary);
                    margin: 2rem 0 1.5rem;
                    padding-bottom: 0.5rem;
                    border-bottom: 2px solid var(--secondary);
                }
        
                /* 内容区块 */
                .feature-section {
                    background: white;
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin: 1.5rem 0;
                    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
                }
        
                .feature-section p {
                    font-size: var(--text-lg);
                    line-height: 1.8;
                    margin: 0.8rem 0;
                }
        
                /* 功能列表 */
                .feature-list li {
                    font-size: var(--text-lg);
                    padding-left: 2rem;
                    margin: 1rem 0;
                    position: relative;
                }
        
                .feature-list li:before {
                    content: "•";
                    color: var(--secondary);
                    font-size: 1.5em;
                    position: absolute;
                    left: 0;
                    top: -0.1em;
                }
        
                /* 引用区块 */
                .quote-section {
                    font-size: var(--text-lg);
                    background: #f8f9fa;
                    border-left: 3px solid var(--secondary);
                    padding: 1.2rem;
                    margin: 1.5rem 0;
                    border-radius: 0 8px 8px 0;
                }
        
                /* 响应式调整 */
                @media (min-width: 768px) {
                    :root {
                        --text-base: 1.2rem;
                        --text-lg: 1.35rem;
                        --text-xl: 1.6rem;
                        --title-sm: 1.9rem;
                        --title-md: 2.2rem;
                        --title-lg: 2.5rem;
                    }
                    
                    .section-title {
                        margin: 2.5rem 0 2rem;
                    }
                }
        
                @media (max-width: 480px) {
                    :root {
                        --text-base: 1.1rem;
                        --title-lg: 2rem;
                    }
                }
            </style>
            """, unsafe_allow_html=True)
        
            # 平台简介
            st.markdown("""
            <div class="feature-section">
                <p>
                    本平台融合AI与材料科学技术，用于可持续高分子复合材料智能设计，重点关注材料阻燃、力学和耐热等性能的优化与调控。
                </p>
            </div>
            """, unsafe_allow_html=True)
        
            st.markdown("""
            <style>
                .feature-list {
                    list-style: none; /* 移除默认列表符号 */
                    padding-left: 0;  /* 移除默认左内边距 */
                }
                .feature-list li:before {
                    content: "•";
                    color: var(--secondary);
                    font-size: 1.5em;
                    position: relative;
                    left: -0.8em;    /* 微调定位 */
                    vertical-align: middle;
                }
                .feature-list li {
                    margin-left: 1.2em;  /* 给符号留出空间 */
                    text-indent: -1em;   /* 文本缩进对齐 */
                }
            </style>
            
            <div class="section-title">核心功能</div>
            <div class="feature-section">
                <ul class="feature-list">
                    <li><strong>性能预测</strong></li>
                    <li><strong>配方建议</strong></li>
                    <li><strong>添加剂推荐</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
            # 研究成果
            st.markdown('<div class="section-title">研究成果</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="quote-section">
                Ma Weibin, Li Ling, Zhang Yu, Li Minjie, Song Na, Ding Peng. <br>
                <em>Active learning-based generative design of halogen-free flame-retardant polymeric composites.</em> <br>
                <strong>J Mater Inf</strong> 2025;5:09. DOI: <a href="http://dx.doi.org/10.20517/jmi.2025.09" target="_blank">10.20517/jmi.2025.09</a>
            </div>
            """, unsafe_allow_html=True)
        
            # 致谢部分
            st.markdown('<div class="section-title">致谢</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="feature-section">
                <p style="font-size: var(--text-lg);">
                    本研究获得云南省科技重点计划项目(202302AB080022)支持
                </p>
            </div>
            """, unsafe_allow_html=True)
        
            # 开发者信息
            st.markdown('<div class="section-title">开发者</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="feature-section">
                <p style="font-size: var(--text-lg);">
                    上海大学功能高分子团队-PolyDesign：马维宾，李凌，张瑜，宋娜，丁鹏
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # 性能预测页面
        elif page == "性能预测":
            st.subheader("🔮 性能预测：基于配方预测LOI和TS")
        
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
        
            fraction_type = st.sidebar.selectbox("选择输入的单位", ["质量", "质量分数", "体积分数"])
        
            st.subheader("请选择配方中的基体、阻燃剂和助剂")
            selected_matrix = st.selectbox("选择基体", matrix_materials, index=0)
            selected_flame_retardants = st.multiselect("选择阻燃剂", flame_retardants, default=["ZS"])
            selected_additives = st.multiselect("选择助剂", additives, default=["wollastonite"])
        
            input_values = {}
            unit_matrix = get_unit(fraction_type)
            unit_flame_retardant = get_unit(fraction_type)
            unit_additive = get_unit(fraction_type)
        
            input_values[selected_matrix] = st.number_input(f"选择 {selected_matrix} ({unit_matrix})", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        
            for fr in selected_flame_retardants:
                input_values[fr] = st.number_input(f"选择 {fr}({unit_flame_retardant})", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        
            for ad in selected_additives:
                input_values[ad] = st.number_input(f"选择 {ad} ({unit_additive})", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        
            total = sum(input_values.values())
            is_only_pp = all(v == 0 for k, v in input_values.items() if k != "PP")
        
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
        
                # 模型验证样本
            with st.expander("📊 模型精度验证"):
                samples = [
                    {
                        "name": "配方1",
                        "配方": {"PP": 63.2, "PAPP": 23.0, "ZS": 1.5, "Anti-drip-agent": 0.3, "MPP": 9.0, "wollastonite": 3.0},
                        "LOI_真实值": 43.5,
                        "TS_真实值": 15.845
                    },
                    {
                        "name": "配方2",
                        "配方": {"PP": 65.2, "PAPP": 23.0, "ZS": 1.5, "Anti-drip-agent": 0.3, "MPP": 7.0, "wollastonite": 3.0},
                        "LOI_真实值": 43.0,
                        "TS_真实值": 16.94
                    },
                    {
                        "name": "配方3",
                        "配方": {"PP": 58.2, "PAPP": 23.0, "ZS": 0.5, "Anti-drip-agent": 0.3, "MPP": 13.0, "wollastonite": 5.0},
                        "LOI_真实值": 43.5,
                        "TS_真实值": 15.303
                    }
                ]
                
                # 设置列布局
                col1, col2, col3 = st.columns(3)
                
                # 循环显示每个配方的内容
                for i, sample in enumerate(samples):
                    with [col1, col2, col3][i]:  # 根据配方编号选择列
                        st.markdown(f"### {sample['name']}")
                        
                        # 显示配方具体内容
                        st.write("配方：")
                        for ingredient, value in sample["配方"].items():
                            st.write(f"  - {ingredient}: {value}wt %")
            
                all_features = set(models["loi_features"]) | set(models["ts_features"])
            
                for sample in samples:
                    # 初始化输入向量（显式包含所有模型特征）
                    input_vector = {feature: 0.0 for feature in all_features}
                    
                    # 填充样本数据
                    for k, v in sample["配方"].items():
                        if k not in input_vector:
                            st.warning(f"检测到样本中存在模型未定义的特征: {k}")
                        input_vector[k] = v  # 存在的特征会被覆盖，不存在的特征会显示警告
            
                    # LOI预测
                    try:
                        loi_input = np.array([[input_vector[f] for f in models["loi_features"]]])
                        loi_scaled = models["loi_scaler"].transform(loi_input)
                        loi_pred = models["loi_model"].predict(loi_scaled)[0]
                    except KeyError as e:
                        st.error(f"LOI模型特征缺失: {e}，请检查模型配置")
                        st.stop()
            
                    # TS预测
                    try:
                        ts_input = np.array([[input_vector[f] for f in models["ts_features"]]])
                        ts_scaled = models["ts_scaler"].transform(ts_input)
                        ts_pred = models["ts_model"].predict(ts_scaled)[0]
                    except KeyError as e:
                        st.error(f"TS模型特征缺失: {e}，请检查模型配置")
                        st.stop()
            
                    loi_error = abs(sample["LOI_真实值"] - loi_pred) / sample["LOI_真实值"] * 100
                    ts_error = abs(sample["TS_真实值"] - ts_pred) / sample["TS_真实值"] * 100
            
                    # 根据误差设置颜色
                    loi_color = "green" if loi_error < 15 else "red"
                    ts_color = "green" if ts_error < 15 else "red"
                    
                    # 显示结果
                    with [col1, col2, col3][samples.index(sample)]:
                        st.markdown(f"""
                        <div class="sample-box">
                            <div class="sample-title">📌 {sample["name"]}</div>
                            <div class="metric-badge" style="color: {loi_color}">LOI误差: {loi_error:.1f}%</div>
                            <div class="metric-badge" style="color: {ts_color}">TS误差: {ts_error:.1f}%</div>
                            <div style="margin-top: 0.8rem;">
                                🔥 真实LOI: {sample["LOI_真实值"]}% → 预测LOI: {loi_pred:.2f}%
                            </div>
                            <div>💪 真实TS: {sample["TS_真实值"]} MPa → 预测TS: {ts_pred:.2f} MPa</div>
                        </div>
                        """, unsafe_allow_html=True)
            
                        if loi_error < 15 and ts_error < 15:
                            st.success(f"✅ {sample['name']}：模型精度超过85%")
                        else:
                            st.warning(f"⚠️ {sample['name']}：模型预测误差较大")
        
            if st.button("🚀 开始预测", type="primary"):
                if fraction_type in ["体积分数", "质量分数"] and abs(total - 100.0) > 1e-6:
                    st.error(f"预测中止：{fraction_type}的总和必须为100%")
                    st.stop()
        
                if is_only_pp:
                    loi_pred = 17.5
                    ts_pred = 35.0
                else:
                    if fraction_type == "体积分数":
                        vol_values = np.array(list(input_values.values()))
                        mass_values = vol_values
                        total_mass = mass_values.sum()
                        input_values = {k: (v / total_mass * 100) for k, v in zip(input_values.keys(), mass_values)}
        
                    for feature in models["loi_features"]:
                        if feature not in input_values:
                            input_values[feature] = 0.0
        
                    loi_input = np.array([[input_values[f] for f in models["loi_features"]]])
                    loi_scaled = models["loi_scaler"].transform(loi_input)
                    loi_pred = models["loi_model"].predict(loi_scaled)[0]
        
                    for feature in models["ts_features"]:
                        if feature not in input_values:
                            input_values[feature] = 0.0
        
                    ts_input = np.array([[input_values[f] for f in models["ts_features"]]])
                    ts_scaled = models["ts_scaler"].transform(ts_input)
                    ts_pred = models["ts_model"].predict(ts_scaled)[0]
        
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="LOI预测值", value=f"{loi_pred:.2f}%")
                with col2:
                    st.metric(label="TS预测值", value=f"{ts_pred:.2f} MPa")
        
        
        
        elif page == "配方建议":
            if sub_page == "配方优化":
                fraction_type = st.sidebar.radio(
                    "📐 单位类型",
                    ["质量", "质量分数", "体积分数"],
                    key="unit_type"
                )
                st.subheader("🧪 配方建议：根据目标LOI和TS优化配方")
            
                matrix_materials = ["PP", "PA", "PC/ABS", "POM", "PBT", "PVC", "其他"]
                flame_retardants = ["AHP", "ammonium octamolybdate", "Al(OH)3", "CFA", "APP", "Pentaerythritol", "DOPO",
                                    "EPFR-1100NT", "XS-FR-8310", "ZS", "XiuCheng", "ZHS", "ZnB", "antimony oxides",
                                    "Mg(OH)2", "TCA", "MPP", "PAPP", "其他"]
                additives = ["Anti-drip-agent", "wollastonite", "M-2200B", "ZBS-PV-OA", "FP-250S", "silane coupling agent", "antioxidant",
                             "SiO2", "其他"]
            
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
