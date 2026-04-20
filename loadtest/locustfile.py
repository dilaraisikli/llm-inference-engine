"""Locust yük testi — 4 user persona.

Çalıştır: locust -f loadtest/locustfile.py --host http://localhost:8000
Sonra: http://localhost:8089 aç

CPU için önerilen ayarlar:
  Number of users: 2
  Spawn rate: 1
"""

import random
from locust import HttpUser, task, between


RESEARCHER_QUERIES = [
    "What optimizer was used for training?",
    "What evaluation metrics were reported?",
    "What dataset was used for experiments?",
    "What is the model architecture?",
    "How was the baseline model defined?",
]

STUDENT_QUERIES = [
    "What is attention mechanism?",
    "How does BERT work?",
    "What is transfer learning?",
    "What is the difference between precision and recall?",
    "What is a transformer model?",
]

CLINICIAN_QUERIES = [
    "What were the main results of the experiments?",
    "What are the limitations of this approach?",
    "How does this compare to previous work?",
    "What future work is suggested?",
    "What is the contribution of this paper?",
]

ADVERSARIAL_QUERIES = [
    "asdf jkl qwerty random noise",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "123456789",
    "?!@#$%",
]


# CPU'da Llama 3 yavaş — wait_time yüksek tutuldu
# Locust UI'da: Number of users = 2, Spawn rate = 1

class ResearcherUser(HttpUser):
    weight = 2
    wait_time = between(60, 120)  # her istekten sonra 1-2 dk bekle

    @task
    def ask(self):
        self.client.post("/ask", json={
            "query": random.choice(RESEARCHER_QUERIES),
            "top_k": 5,
            "tags": ["loadtest", "researcher"],
        }, timeout=300)


class StudentUser(HttpUser):
    weight = 2
    wait_time = between(90, 150)

    @task
    def ask(self):
        self.client.post("/ask", json={
            "query": random.choice(STUDENT_QUERIES),
            "top_k": 3,
            "tags": ["loadtest", "student"],
        }, timeout=300)


class ClinicalUser(HttpUser):
    weight = 1
    wait_time = between(60, 120)

    @task(3)
    def ask(self):
        self.client.post("/ask", json={
            "query": random.choice(CLINICIAN_QUERIES),
            "top_k": 5,
            "tags": ["loadtest", "clinician"],
        }, timeout=300)

    @task(1)
    def health(self):
        self.client.get("/health")


class AdversarialUser(HttpUser):
    weight = 1
    wait_time = between(120, 180)

    @task
    def ask(self):
        with self.client.post("/ask", json={
            "query": random.choice(ADVERSARIAL_QUERIES),
            "top_k": 5,
            "tags": ["loadtest", "adversarial"],
        }, timeout=300, catch_response=True) as response:
            if response.status_code in (400, 422):
                response.success()