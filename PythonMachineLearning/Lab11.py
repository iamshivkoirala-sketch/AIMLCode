import os
import openai

api_key = os.getenv("aikey")
openai.api_key = api_key

def generate_interview_questions(topic):
    """
    Generate 10 interview questions around a given topic
    """
    messages = [
        {
            "role": "system",
            "content": f"You are an expert interviewer. Create 10 interview questions about {topic}. List them clearly."
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # or "gpt-3.5-turbo"
        messages=messages
    )

    questions = response.choices[0].message['content']
    return questions

def main():
    topic = input("Which interview do you want to take? (e.g., C#, Python, Data Science): ")
    print(f"\nGenerating 10 interview questions on: {topic}...\n")

    questions = generate_interview_questions(topic)
    print(questions)

if __name__ == "__main__":
    main()
