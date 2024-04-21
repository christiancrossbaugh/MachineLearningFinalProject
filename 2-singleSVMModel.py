import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from nltk.corpus import wordnet
import random

nltk.download('wordnet')

# Function for synonym replacement data augmentation
def synonym_replacement(text, n=2):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stopwords.words('english')]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

# Step 1: Read CSV and preprocess data
df = pd.read_csv('/content/software_req_classified_raw.csv', delimiter=',')
# Assuming the CSV contains two columns: 'requirement' and 'reqLabel'
X = df['requirement']
y = df['reqLabel']

# Determine the minority class
minority_class = y.value_counts().idxmin()

# Data augmentation
augmented_X = []
augmented_y = []
for text, label in zip(X, y):
    augmented_X.append(text)
    augmented_y.append(label)
    # Apply data augmentation to only the minority class
    if label == minority_class:
        augmented_text = synonym_replacement(text)  # Use other data augmentation techniques as well
        augmented_X.append(augmented_text)
        augmented_y.append(label)

X = augmented_X
y = augmented_y

# Step 2: Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features as needed
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Step 4: Train SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Step 5: Model evaluation
y_pred = svm_classifier.predict(X_test)
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
new_requirements_augmented = [synonym_replacement(req) for req in new_requirements]
new_requirements_tfidf = tfidf_vectorizer.transform(new_requirements_augmented)
predictions = svm_classifier.predict(new_requirements_tfidf)
print("Predictions for new requirements:")
for req, pred in zip(new_requirements, predictions):
    print(f"{req} - Predicted: {pred}")
