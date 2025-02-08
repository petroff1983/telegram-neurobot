import requests

TELEGRAM_BOT_TOKEN = "8022323485:AAEfenu6pXnhvig1fn-HMvnmxmO9zQPSBJY"

# Удаляем Webhook перед запуском Polling
requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook")

import os
import logging
import openai
import asyncio
import faiss
import pickle
import shutil
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import CommandStart
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Настройки бота
TELEGRAM_BOT_TOKEN = "8022323485:AAEfenu6pXnhvig1fn-HMvnmxmO9zQPSBJY"
OPENAI_API_KEY = "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"

# Проверяем, загружен ли ключ
if not OPENAI_API_KEY:
    raise ValueError(
        "Отсутствует OPENAI_API_KEY. Добавьте его в переменные Railway.")

# Устанавливаем ключ API OpenAI
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Логирование
logging.basicConfig(level=logging.INFO)

# Загрузка FAISS-индекса
INDEX_FOLDER = "faiss_index"
INDEX_ZIP = "faiss_index.zip"

if os.path.exists(INDEX_ZIP):
    shutil.unpack_archive(INDEX_ZIP, INDEX_FOLDER)
    vector_store = FAISS.load_local(INDEX_FOLDER, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
else:
    print("⚠️ FAISS-индекс не найден, создаю пустой векторный стор.")
    vector_store = None

# Функция обработки команды /start


@dp.message(CommandStart())
async def start_handler(message: Message):
    await message.answer("Привет! Я нейроконсультант 🤖. Задавай вопросы, и я помогу!")

# Функция обработки текстовых сообщений


@dp.message()
async def process_message(message: Message):
    user_input = message.text
    response = ask_ai(user_input)
    await message.answer(response)

# Функция запроса к FAISS и OpenAI


def ask_ai(query: str) -> str:
    global vector_store
    if vector_store:
        docs = vector_store.similarity_search(query, k=2)
        context = "\n".join([doc.page_content for doc in docs])
        query = f"Контекст:\n{context}\n\nВопрос: {query}"

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": query}]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Ошибка при запросе к OpenAI: {e}"

# Запуск бота


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
