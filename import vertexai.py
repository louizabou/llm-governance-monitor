import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project='i-destiny-461017-g2', location='us-central1')
model = GenerativeModel('gemini-2.0-flash-001')
response = model.generate_content('Hello')
print(response.text)