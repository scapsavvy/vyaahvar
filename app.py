import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from nrclex import NRCLex
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import streamlit as st
import time
import matplotlib.pyplot as plt
import seaborn as sns
import PyPDF2
from PIL import Image
import pytesseract
from wordcloud import WordCloud

# Ensure required nltk resources are downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
# Ensure necessary NLTK data is downloaded
nltk.download('punkt', quiet=True)

# Set up page configuration
st.set_page_config(page_title="Vyaahvar Drishti", layout="wide")

# Load custom CSS styles
def load_css():
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&family=Poppins:wght@400;700&display=swap');

    body {
        font-family: 'Montserrat', sans-serif;
        background: linear-gradient(135deg, #e0f7fa, #c7d2fe 100%);
        color: #333;
        margin: 0;
        padding: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100vh;
    }
    .container {
        text-align: center;
        padding: 20px;
        margin-top: 50px;
    }
    .header img {
        width: 250px;
        margin-bottom: 20px;
    }
    .header h1 {
        font-family: 'Poppins', sans-serif;
        color: #004d40;
        font-size: 56px;
        margin: 20px 0;
        animation: fadeIn 2s ease-in-out;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .header p {
        font-family: 'Montserrat', sans-serif;
        color: #333;
        font-size: 22px;
        margin: 10px 0;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        font-size: 16px;
        margin-top: 30px;
        color: #555;
        background: rgba(255, 255, 255, 0.8);
        padding: 10px 0;
    }
    .button {
        background-color: #00796b;
        color: white;
        border: none;
        padding: 15px 30px;
        font-size: 18px;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s ease;
        margin-top: 20px;
        font-family: 'Poppins', sans-serif;
    }
    .button:hover {
        background-color: #004d40;
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    """
    st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)

# Load styles
load_css()

# Sidebar navigation
st.sidebar.title("Navigation")
menu = ["🏠Home", "📊Generate Report", "📞Contact Us"]
option = st.sidebar.selectbox("Navigation", menu)

# Home Section
if option == "🏠Home":  # Ensure the option matches the menu name
    st.markdown(
        """
        <div style="
            text-align: center; 
            padding: 20px; 
            margin-top: 50px; 
            font-family: 'Poppins', sans-serif; 
            background: linear-gradient(135deg, #e0f7fa, #c7d2fe); 
            color: #333; 
            border-radius: 15px; 
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h1 style="color: #004d40; font-size: 56px; margin: 20px 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);">
                🌟NUVLeap🌟
            </h1>
            <p style="font-size: 22px; margin: 10px 0;">
                Welcome to the <strong>NUVLeap platform</strong>, where insights meet innovation! Discover organizational behavior analysis and explore in-depth metrics to unlock the potential of your company.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


    # Display the image using st.image
    st.image("https://i.imgur.com/2JWfPlk.png", width=250, caption="NUVLeap Logo")

# Sentiment analysis using TextBlob
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity
    
# Preprocess text for TF-IDF and keyword analysis
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)
    
def analyze_emotion(text):
    preprocessed_text = preprocess_text(text)
    emotion_scores = NRCLex(preprocessed_text).affect_frequencies
    emotions = {emotion: round(score, 2) for emotion, score in emotion_scores.items() if emotion not in ['positive', 'negative']}
    dominant_emotion = max(emotions, key=emotions.get, default="none")
    return {
        "Dominant Emotion": dominant_emotion,
        "Emotion Scores": emotions
    }

# Aggregate emotion scores from multiple reviews
def aggregate_emotion_scores(reviews):
    aggregated_emotions = {}
    for review in reviews:
        emotion_results = analyze_emotion(review)
        for emotion, score in emotion_results["Emotion Scores"].items():
            aggregated_emotions[emotion] = aggregated_emotions.get(emotion, 0) + score
    dominant_emotion = max(aggregated_emotions, key=aggregated_emotions.get, default="none")
    return dominant_emotion, aggregated_emotions

# Alignment scoring using cosine similarity
def calculate_alignment_score(internal_clusters, external_data, num_clusters=5):
    """Calculate alignment score between internal clusters and external data."""
    vectorizer = TfidfVectorizer()
    internal_tfidf_matrix = vectorizer.fit_transform(internal_clusters)
    external_tfidf_matrix = vectorizer.transform(external_data)

    # Calculate centroids for internal clusters
    cluster_centroids = []
    for i in range(num_clusters):
        # Use modulo operation on index positions to group into clusters
        cluster_indices = [index for index in range(len(internal_clusters)) if index % num_clusters == i]
        cluster_centroid = internal_tfidf_matrix[cluster_indices].mean(axis=0)
        cluster_centroids.append(cluster_centroid)
    cluster_centroids = np.array(cluster_centroids).reshape(num_clusters, -1)

    # Cosine similarity between external data and cluster centroids
    external_similarity = cosine_similarity(external_tfidf_matrix, cluster_centroids)
    alignment_score = external_similarity.max(axis=1).mean()

    return alignment_score

# Load text data from a file
def load_data(file):
    if file.name.endswith('.txt'):
        return [line.strip() for line in file.getvalue().decode('utf-8').splitlines()]
    elif file.name.endswith('.csv'):
        df = pd.read_csv(file)
        return df.iloc[:, 0].tolist()  # Assuming the text data is in the first column
    elif file.name.endswith('.pdf'):
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extract_text()
        return text.splitlines()
    elif file.name.endswith('.png') or file.name.endswith('.jpg') or file.name.endswith('.jpeg'):
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
        return text.splitlines()
    else:
        st.error("Unsupported file format. Please upload a .txt, .csv, .pdf, .png, .jpg, or .jpeg file.")
        return []

# Analyze sentiment (using TextBlob)
def analyze_sentiment(reviews):
    sentiments = [get_sentiment(review) for review in reviews]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    return avg_sentiment

# Count mentions (keywords related to suppliers, regulators, etc.)
def count_mentions(reviews, keyword):
    return sum(keyword in review for review in reviews)

# Generate diagnostics based on internal and external data
def generate_diagnostics(company_name, alignment_score, avg_external_sentiment, dominant_external_emotion, avg_internal_sentiment, supplier_mentions, regulator_mentions):
    # Adjusted thresholds for sentiment classification
    if avg_external_sentiment > 0.5:
        sentiment_label = 'Very Positive'
    elif avg_external_sentiment > 0:
        sentiment_label = 'Less Positive'
    elif avg_external_sentiment > -0.5:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Strongly Negative'

    # Customer feedback (external data) sentiment statement
    customer_feedback = f"{sentiment_label} feedback on ethical behavior, with an average sentiment score of {avg_external_sentiment:.2f}."

    # Supplier fairness statement
    supplier_fairness = f"Fair treatment observed in {supplier_mentions} supplier-related mentions." if supplier_mentions > 0 else "No significant mentions of supplier fairness."

    # Regulator mentions statement
    regulator_mentions_statement = f"Regulator-related mentions observed in {regulator_mentions} instances." if regulator_mentions > 0 else "No significant mentions of regulator-related issues."

    # Employee engagement (internal data) sentiment statement
    if avg_internal_sentiment > 0.5:
        employee_engagement = f"Employee engagement is exceptionally positive with a sentiment score of {avg_internal_sentiment:.2f}, indicating high levels of satisfaction and enthusiasm."
    elif avg_internal_sentiment > 0:
        employee_engagement = f"Employee engagement is less positive with a sentiment score of {avg_internal_sentiment:.2f}, suggesting a generally favorable work environment and morale."
    elif avg_internal_sentiment > -0.5:
        employee_engagement = f"Employee engagement shows room for improvement with a sentiment score of {avg_internal_sentiment:.2f}, highlighting areas that may need attention and support."
    else:
        employee_engagement = f"Employee engagement is notably negative with a sentiment score of {avg_internal_sentiment:.2f}, reflecting significant dissatisfaction and potential concerns."

    # Return the generated diagnostics
    return {
        "Company Name": company_name,
        "Customer Feedback": customer_feedback,
        "Supplier Fairness": supplier_fairness,
        "Regulator Mentions": regulator_mentions_statement,
        "Employee Engagement": employee_engagement,
        "Alignment Score": f" {alignment_score:.2f}",
        "Dominant External Emotion": f" {dominant_external_emotion}"
    }

# Generate prognostics
def generate_prognostics(alignment_score, avg_internal_sentiment):
    if alignment_score > 0.75 and avg_internal_sentiment > 0.5:
        return "Positive. The company is well-aligned with its values and stakeholders."
    elif alignment_score > 0.5 or avg_internal_sentiment > 0.3:
        return "Neutral. There is room for improvement in aligning company values with stakeholder actions."
    else:
        return "Negative. Significant misalignment between stated values and stakeholder actions."

# Generate risk, opportunity, and improvement insights
def generate_risk_opportunity_improvement(alignment_score, avg_internal_sentiment, dominant_external_emotion):
    if alignment_score < 0.5:
        risk = "Significant misalignment may affect long-term trust and reputation."
    else:
        risk = "Minor misalignment, which can be addressed."

    if avg_internal_sentiment < 0.3:
        opportunity = "Internal sentiment is low. Focus on improving employee engagement."
    else:
        opportunity = "The moderately positive internal sentiment presents an opportunity to strengthen alignment with core values."

    if dominant_external_emotion == "fear":
        improvement = "Improve transparency and build trust."
    elif dominant_external_emotion == "anger":
        improvement = "Address concerns through open dialogues."
    else:
        improvement = "Maintain positive relationships with stakeholders."

    return risk, opportunity, improvement

# Generate prescriptive analytics
def generate_prescriptive_analytics(alignment_score, avg_internal_sentiment, dominant_external_emotion):
    improvement_strategies = []

    if alignment_score < 0.5:
        improvement_strategies.append("Implement a comprehensive alignment strategy to bridge the gap between company values and stakeholder actions.")
    else:
        improvement_strategies.append("Continue to reinforce alignment initiatives to maintain strong relationships with stakeholders.")

    if avg_internal_sentiment < 0.3:
        improvement_strategies.append("Focus on employee engagement programs to boost morale and satisfaction.")
    else:
        improvement_strategies.append("Sustain and enhance employee engagement initiatives to foster a positive work environment.")

    if dominant_external_emotion == "fear":
        improvement_strategies.append("Address customer fears by increasing transparency and building trust through open communication.")
    elif dominant_external_emotion == "anger":
        improvement_strategies.append("Address customer anger by actively listening to their concerns and taking corrective actions.")
    else:
        improvement_strategies.append("Maintain positive customer relationships by continuing to deliver high-quality products and services.")

    return improvement_strategies

# Generate word cloud
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white',scale=2).generate(text)
    plt.figure(figsize=(5, 3.5))  # Adjusted size
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Generate sentiment distribution chart
def generate_sentiment_distribution_chart(sentiments):
    sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    for sentiment in sentiments:
        if sentiment > 0:
            sentiment_counts['Positive'] += 1
        elif sentiment == 0:
            sentiment_counts['Neutral'] += 1
        else:
            sentiment_counts['Negative'] += 1

    sentiment_df = pd.DataFrame(list(sentiment_counts.items()), columns=['Sentiment', 'Count'])
    sns.set(style="whitegrid")
    plt.figure(figsize=(5, 3.5))  # Adjusted size
    sns.barplot(x='Sentiment', y='Count', data=sentiment_df, palette="viridis")
    plt.title('Sentiment Distribution', fontsize=12)
    plt.xlabel('Sentiment', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    st.pyplot(plt)

# Report generation section
if option == "Generate Report":
    st.title("Generate Vyaahvar Drishti Report")
    company_name = st.text_input("Enter the company name to generate a report:")

    # File uploaders for internal, external, and employee reviews data
    internal_data_file = st.file_uploader("Upload internal data file (e.g., internal_lda_lexicon.txt, .csv, .pdf, .png, .jpg, .jpeg)", type=["txt", "csv", "pdf", "png", "jpg", "jpeg"])
    external_data_file = st.file_uploader("Upload external data file (e.g., ext_cleaned_coca_cola_reviews.txt, .csv, .pdf, .png, .jpg, .jpeg)", type=["txt", "csv", "pdf", "png", "jpg", "jpeg"])
    employee_reviews_file = st.file_uploader("Upload employee reviews file (e.g., int_cleaned_coca_cola_reviews.txt, .csv, .pdf, .png, .jpg, .jpeg)", type=["txt", "csv", "pdf", "png", "jpg", "jpeg"])

    # File validation for proper uploads
    def validate_files(files):
        for file in files:
            if file is not None:
                file_extension = file.name.split('.')[-1]
                if file_extension not in ["txt", "csv", "pdf", "png", "jpg", "jpeg"]:
                    return False, f"Invalid file type for {file.name}. Please upload a .txt, .csv, .pdf, .png, .jpg, or .jpeg file."
        return True, ""

    if st.button("Generate Report"):
        # Check that all files are uploaded and valid
        files = [internal_data_file, external_data_file, employee_reviews_file]
        valid, error_message = validate_files(files)

        if valid and company_name and internal_data_file and external_data_file and employee_reviews_file:
            with st.spinner(f"Generating report for {company_name}..."):
                time.sleep(3)  # Simulate processing

                # Load data from uploaded files
                internal_data = load_data(internal_data_file)
                external_data = load_data(external_data_file)
                employee_reviews = load_data(employee_reviews_file)

                if not internal_data or not external_data or not employee_reviews:
                    st.error("Failed to load data. Please check the uploaded files.")
                else:
                    # Preprocess external data
                    processed_external_data = [preprocess_text(text) for text in external_data]

                    # Calculate alignment score
                    alignment_score = calculate_alignment_score(internal_data, processed_external_data)

                    # Internal sentiment analysis
                    internal_sentiments = [get_sentiment(text) for text in employee_reviews]
                    avg_internal_sentiment = sum(internal_sentiments) / len(internal_sentiments)

                    # External sentiment analysis
                    external_sentiments = [get_sentiment(text) for text in external_data]
                    avg_external_sentiment = sum(external_sentiments) / len(external_sentiments)

                    # External emotion analysis
                    dominant_external_emotion, aggregated_emotions = aggregate_emotion_scores(external_data)

                    # Count mentions of supplier and regulator-related terms
                    supplier_keywords = ['supplier', 'fairness', 'quality', 'delivery', 'partnership']
                    regulator_keywords = ['regulator', 'compliance', 'law', 'policy', 'regulation']
                    supplier_mentions = sum([1 for text in processed_external_data if any(keyword in text for keyword in supplier_keywords)])
                    regulator_mentions = sum([1 for text in processed_external_data if any(keyword in text for keyword in regulator_keywords)])

                    # Generate diagnostic report
                    diagnostics = generate_diagnostics(
                        company_name=company_name,
                        alignment_score=alignment_score,
                        avg_external_sentiment=avg_external_sentiment,
                        dominant_external_emotion=dominant_external_emotion,
                        avg_internal_sentiment=avg_internal_sentiment,
                        supplier_mentions=supplier_mentions,
                        regulator_mentions=regulator_mentions
                    )

                    # Generate prognostic report
                    prognostics = generate_prognostics(alignment_score, avg_internal_sentiment)

                    # Generate risk, opportunity, and improvement report
                    risk, opportunity, improvement = generate_risk_opportunity_improvement(alignment_score, avg_internal_sentiment, dominant_external_emotion)

                    # Generate prescriptive analytics
                    improvement_strategies = generate_prescriptive_analytics(alignment_score, avg_internal_sentiment, dominant_external_emotion)

                    # Display the results
                    st.success(f"Report for {company_name}")
                    st.subheader("Diagnostics")
                    for key, value in diagnostics.items():
                        st.markdown(f"<div class='card'><p style='font-size: 18px; color: #333;'><strong>{key}:</strong> {value}</p></div>", unsafe_allow_html=True)

                    st.subheader("Prognostics")
                    st.write(prognostics)

                    st.subheader("Risk, Opportunity, and Improvement")
                    st.write(f"**Risk:** {risk}")
                    st.write(f"**Opportunity:** {opportunity}")
                    st.write(f"**Improvement:** {improvement}")

                    st.subheader("Prescriptive Analytics")
                    for strategy in improvement_strategies:
                        st.write(f"- {strategy}")

                    st.subheader("Emotion Analysis")
                    emotion_data = aggregated_emotions
                    emotion_df = pd.DataFrame(list(emotion_data.items()), columns=['Emotion', 'Score'])
                    sns.set(style="whitegrid")
                    plt.figure(figsize=(5, 3.5))  # Adjusted size
                    sns.barplot(x='Score', y='Emotion', data=emotion_df, palette="coolwarm")
                    plt.title('Emotion Analysis', fontsize=12)
                    plt.xlabel('Score', fontsize=10)
                    plt.ylabel('Emotion', fontsize=10)
                    plt.xticks(fontsize=8)
                    plt.yticks(fontsize=8)
                    st.pyplot(plt)

                    st.subheader("Word Cloud - Internal Data")
                    generate_word_cloud(' '.join(internal_data))
                    st.subheader("Word Cloud - External Data")
                    generate_word_cloud(' '.join(external_data))

                    # Generate sentiment distribution chart
                    st.subheader("Sentiment Distribution - Internal Data")
                    generate_sentiment_distribution_chart(internal_sentiments)
                    st.subheader("Sentiment Distribution - External Data")
                    generate_sentiment_distribution_chart(external_sentiments)
        else:
            st.error(error_message or "Please enter a company name and upload all required files to generate a report.")

# Contact Us Section
elif option == "Contact Us":
    st.subheader("Contact Us")
    st.write("If you have any questions or feedback, please reach out to us at:")
    st.write("📧 Email: info@nuvleap.com")
    st.write("📞 Phone: +91 77100 63035")
    st.write("🌐 Website: [NUVLeap Official Website](https://nuvleap.com)")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='footer'>Developed with ❤️ by NUVLeap Team</div>", unsafe_allow_html=True)
