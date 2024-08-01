import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained model and tokenizer
model_name = "valhalla/t5-small-qg-hl"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to generate questions from text
def generate_questions(text):
    input_text = "generate questions: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids=input_ids, max_length=512, num_beams=4, early_stopping=True)
    questions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return questions

# Function to evaluate the provided answers
def evaluate_answers(questions, user_answers):
    # Placeholder logic for answer evaluation
    # In practice, you would have a way to validate answers (e.g., using a model or pre-stored answers)
    correct_answers = [f"This is the correct answer to question {i + 1}" for i in range(len(questions))]
    evaluations = []
    for idx, answer in enumerate(user_answers):
        if answer.lower().strip() == correct_answers[idx].lower().strip():
            evaluations.append("Correct")
        else:
            evaluations.append(f"Incorrect. The correct answer is: {correct_answers[idx]}")
    return evaluations

# Streamlit app
def main():
    st.title("Document-based Question Generation and Evaluation")

    # Sidebar for file upload
    uploaded_file = st.file_uploader("Upload a document:", type=["pdf"])

    if uploaded_file is not None:
        # Read file content
        text = uploaded_file.read().decode("utf-8")

        # Generate questions
        st.header("Generated Questions")
        questions = generate_questions(text)

        # Input fields for user answers
        user_answers = []
        for idx, question in enumerate(questions):
            st.write(f"Question {idx + 1}: {question}")
            user_answer = st.text_input(f"Your Answer for Question {idx + 1}:", key=f"answer_{idx}")
            user_answers.append(user_answer)
        
        if st.button("Evaluate Answers"):
            # Evaluate the provided answers
            evaluations = evaluate_answers(questions, user_answers)
            for idx, evaluation in enumerate(evaluations):
                st.write(f"Question {idx + 1}: {evaluation}")

if __name__ == "__main__":
    main()
