{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ce19643d-c62b-46c6-9a1d-25ca361d4acc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "* Running on local URL:  http://127.0.0.1:7861\n",
      "* To create a public link, set `share=True` in `launch()`.\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div><iframe src=\"http://127.0.0.1:7861/\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": []
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import json\n",
    "import gradio as gr\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "\n",
    "# =========================\n",
    "# LOAD FAQ DATA\n",
    "# =========================\n",
    "with open(\"faqs.json\", \"r\", encoding=\"utf-8\") as f:\n",
    "    data = json.load(f)\n",
    "\n",
    "questions = [item[\"question\"] for item in data]\n",
    "answers = [item[\"answer\"] for item in data]\n",
    "\n",
    "# =========================\n",
    "# TRAIN SIMPLE NLP MODEL\n",
    "# =========================\n",
    "vectorizer = TfidfVectorizer()\n",
    "question_vectors = vectorizer.fit_transform(questions)\n",
    "\n",
    "# =========================\n",
    "# CHATBOT RESPONSE FUNCTION\n",
    "# =========================\n",
    "def chatbot_response(user_input):\n",
    "\n",
    "    user_vec = vectorizer.transform([user_input.lower()])\n",
    "    similarity = cosine_similarity(user_vec, question_vectors)\n",
    "\n",
    "    best_index = similarity.argmax()\n",
    "    confidence = similarity[0][best_index]\n",
    "\n",
    "    # small talk responses\n",
    "    text = user_input.lower()\n",
    "    if text in [\"hi\",\"hello\",\"hey\"]:\n",
    "        return \"Hello 👋 How can I help you with AI today?\"\n",
    "\n",
    "    if text in [\"how are you\",\"how r u\"]:\n",
    "        return \"I am an AI chatbot and working perfectly 😄\"\n",
    "\n",
    "    # FAQ match\n",
    "    if confidence > 0.35:\n",
    "        return f\"{answers[best_index]}\\n\\n(confidence: {confidence:.2f})\"\n",
    "\n",
    "    return \"Sorry, I only answer AI related questions for now.\"\n",
    "\n",
    "# =========================\n",
    "# GRADIO GUI\n",
    "# =========================\n",
    "demo = gr.Interface(\n",
    "    fn=chatbot_response,\n",
    "    inputs=gr.Textbox(placeholder=\"Ask anything about AI...\"),\n",
    "    outputs=\"text\",\n",
    "    title=\"AI FAQ Chatbot - Apexcify Internship\",\n",
    "    description=\"Ask questions related to Artificial Intelligence, Machine Learning, NLP, etc.\"\n",
    ")\n",
    "\n",
    "demo.launch()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "655a6d69-9d4f-4e92-943c-fdac79f0099f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
