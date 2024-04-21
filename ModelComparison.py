import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Step 1: Read CSV and preprocess data
df = pd.read_csv('/content/software_req_classified_raw.csv', delimiter=',')
# Assuming the CSV contains two columns: 'requirement' and 'reqLabel'
X = df['requirement']
y = df['reqLabel']

# Step 2: Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features as needed
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Step 4: Train different classifiers
# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Multinomial Naive Bayes Classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Logistic Regression Classifier
lr_classifier = LogisticRegression(max_iter=1000)
lr_classifier.fit(X_train, y_train)

# Step 5: Model evaluation
classifiers = {
    'Random Forest': rf_classifier,
    'Multinomial Naive Bayes': nb_classifier,
    'Logistic Regression': lr_classifier
}

for name, classifier in classifiers.items():
    print(f"\n{name}:")
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))

# Step 6: Prediction example
new_requirements = ["The system must be able to handle 1000 concurrent users.",
                    "The user interface should be intuitive and easy to use.",
                    "As an administrative user, I need the capability to include new students into the system.",
                    "As a faculty member, it's essential for me to access student records to review their academic journey.",
                    "As a student advisor, I require the ability to schedule appointments with students to provide guidance.",
                    "As a student, I wish to check my class timetable and room allocations conveniently.",
                    "As a registrar, it's imperative for me to enroll students in courses and update their registration status.",
                    "As a financial aid officer, I need to manage student financial aid applications and disbursements efficiently.",
                    "As a department chair, I aim to review and authorize course proposals submitted by faculty members.",
                    "As a lecturer, I want to upload course materials and assignments seamlessly for my students.",
                    "As a librarian, I aim to oversee the university's library resources and monitor student borrowing.",
                    "As a student, I desire access to online resources such as e-books and academic journals.",
                    "As an admissions officer, it's crucial to process student applications and monitor their admission status.",
                    "As a student, I want to register for classes online and track my course history.",
                    "As a faculty member, I need to submit grades for my students promptly at the end of each semester.",
                    "As an IT administrator, I need to ensure the security and integrity of the student management system.",
                    "As a student, I expect to receive timely notifications regarding important deadlines and events.",
                    "As a student, I seek convenience in requesting official transcripts and other documents online.",
                    "As an academic advisor, I require access to students' academic records to offer tailored guidance.",
                    "As a student, I aim to update my personal details, such as contact information and major, easily.",
                    "As a faculty member, I want to generate class rosters and attendance reports effortlessly.",
                    "As a department administrator, I seek to manage faculty assignments and teaching schedules efficiently.",
                    "As a student, I desire easy access to internship and research opportunities available through the system.",
                    "As a university staff member, I want to generate reports on student demographics and enrollment trends accurately.",
                    "As a student, I aim to handle tuition fees and review my financial statement conveniently.",
                    "As a faculty member, I need a seamless means to communicate with my students through messaging or announcements.",
                    "As a student, I want to access academic advising services and schedule appointments conveniently.",
                    "As a university administrator, I need to monitor student retention rates and graduation outcomes effectively.",
                    "As a student, I seek easy access to online tutoring and academic support services."]
for name, classifier in classifiers.items():
    print(f"\nPredictions for new requirements using {name}:")
    new_requirements_tfidf = tfidf_vectorizer.transform(new_requirements)
    predictions = classifier.predict(new_requirements_tfidf)
    for req, pred in zip(new_requirements, predictions):
        print(f"{req} - Predicted: {pred}")