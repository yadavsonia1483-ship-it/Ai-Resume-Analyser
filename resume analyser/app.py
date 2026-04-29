import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Page configuration
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------

st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#1f4037,#99f2c8);
}

.main-title{
font-size:48px;
font-weight:800;
text-align:center;
color:white;
animation: fadeIn 2s ease-in;
}

.subtitle{
text-align:center;
font-size:20px;
color:#f0f0f0;
margin-bottom:30px;
}

.card{
padding:25px;
border-radius:15px;
background:rgba(255,255,255,0.95);
box-shadow:0px 8px 20px rgba(0,0,0,0.2);
transition:transform 0.3s;
}

.card:hover{
transform:scale(1.03);
}

.stButton>button{
background:linear-gradient(90deg,#ff6a00,#ee0979);
color:white;
font-size:18px;
border:none;
border-radius:10px;
padding:10px 25px;
transition:0.3s;
}

.stButton>button:hover{
transform:scale(1.05);
}

.footer{
text-align:center;
color:white;
margin-top:40px;
font-size:16px;
}

@keyframes fadeIn{
from{opacity:0;}
to{opacity:1;}
}

</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------

st.markdown('<p class="main-title">🤖 AI Resume Job Match Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze how well your resume matches a job description using AI</p>', unsafe_allow_html=True)

st.divider()

# ------------------ SIDEBAR ------------------

with st.sidebar:

    st.header("📌 About")

    st.write("""
This AI tool compares your **resume** with a **job description**
and calculates a **match score** using Machine Learning.

Technology Used:
- TF-IDF
- Cosine Similarity
- Natural Language Processing
""")

    st.header("⚙ Steps")

    st.write("""
1️⃣ Upload Resume  
2️⃣ Paste Job Description  
3️⃣ Click Analyze  
4️⃣ View Match Score
""")

# ------------------ FUNCTIONS ------------------

def extract_text_from_pdf(file):

    pdf_reader = PyPDF2.PdfReader(file)

    text = ""

    for page in pdf_reader.pages:

        if page.extract_text():
            text += page.extract_text()

    return text


def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text


def remove_stopwords(text):

    stop_words = set(stopwords.words('english'))

    words = word_tokenize(text)

    return " ".join([w for w in words if w not in stop_words])


def calculate_similarity(resume, job):

    resume = remove_stopwords(clean_text(resume))

    job = remove_stopwords(clean_text(job))

    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform([resume, job])

    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0] * 100

    return round(score, 2)

# ------------------ INPUT SECTION ------------------

col1, col2 = st.columns(2)

with col1:

    
    uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

    st.markdown('</div>', unsafe_allow_html=True)

with col2:

  

    job_description = st.text_area("💼 Paste Job Description", height=200)

    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# ------------------ ANALYZE BUTTON ------------------

if st.button("🚀 Analyze Resume", use_container_width=True):

    if not uploaded_file:

        st.warning("Please upload your resume")

    elif not job_description:

        st.warning("Please paste job description")

    else:

        with st.spinner("Analyzing resume..."):

            resume_text = extract_text_from_pdf(uploaded_file)

            score = calculate_similarity(resume_text, job_description)

# ------------------ RESULTS ------------------

            st.divider()

            st.subheader("📊 Resume Match Score")

            st.metric("Match Score", f"{score}%")

            st.progress(int(score))

# ------------------ CHART ------------------

            fig, ax = plt.subplots()

            colors = ["#ff4b5c","#ffa500","#00c853"]

            color_index = min(int(score // 33), 2)

            ax.barh(["Match Score"], [score], color=colors[color_index])

            ax.set_xlim(0, 100)

            st.pyplot(fig)

# ------------------ FEEDBACK ------------------

            if score < 40:

                st.error("❌ Low Match — Improve resume keywords")

            elif score < 70:

                st.warning("⚠ Moderate Match — Resume partially fits job")

            else:

                st.success("✅ Excellent Match — Resume strongly matches job!")

# ------------------ FOOTER ------------------

st.markdown('<div class="footer">🚀 AI Resume Analyzer | Built with Streamlit</div>', unsafe_allow_html=True)