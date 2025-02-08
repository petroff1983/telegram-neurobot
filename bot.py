import requests
import os
import logging
import openai
import asyncio
import shutil
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import CommandStart
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Загружаем токены из переменных окружения
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Проверяем, загружены ли ключи
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("❌ Отсутствует TELEGRAM_BOT_TOKEN. Добавьте его в переменные Railway.")
if not OPENAI_API_KEY:
    raise ValueError("❌ Отсутствует OPENAI_API_KEY. Добавьте его в переменные Railway.")

# Удаляем Webhook перед запуском Polling
requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook")

# Устанавливаем ключ API OpenAI
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Логирование
logging.basicConfig(level=logging.INFO)

# Читаем базу знаний
KNOWLEDGE_FILE = "knowledge.txt"
INSTRUCTION_FILE = "instruction.txt"

if os.path.exists(KNOWLEDGE_FILE):
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        knowledge = f.read()
else:
    knowledge = "⚠️ ВНИМАНИЕ: База знаний не загружена!"

# Читаем инструкцию
if os.path.exists(INSTRUCTION_FILE):
    with open(INSTRUCTION_FILE, "r", encoding="utf-8") as f:
        instruction = f.read()
else:
    instruction = "⚠️ ВНИМАНИЕ: Инструкция не загружена!"

# Загрузка FAISS-индекса
INDEX_FOLDER = "faiss_index"
INDEX_ZIP = "faiss_index.zip"

vector_store = None
if os.path.exists(INDEX_ZIP):
    shutil.unpack_archive(INDEX_ZIP, INDEX_FOLDER)
    try:
        vector_store = FAISS.load_local(INDEX_FOLDER, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        logging.info("✅ FAISS-индекс загружен успешно.")
    except Exception as e:
        logging.error(f"⚠️ Ошибка при загрузке FAISS-индекса: {e}")
else:
    logging.warning("⚠️ FAISS-индекс не найден. Бот будет работать без контекста.")

# Функция обработки команды /start
@dp.message(CommandStart())
async def start_handler(message: Message):
    await message.answer("Привет! Я нейроконсультант 🤖. Задавай вопросы, и я помогу!")

# Функция обработки текстовых сообщений
@dp.message(lambda message: message.text)
async def process_message(message: Message):
    user_input = message.text
    response = ask_ai(user_input)
    await message.answer(response)

# Функция запроса к FAISS и OpenAI
def ask_ai(query: str) -> str:
    global vector_store
    context = ""

    # ОГРАНИЧИВАЕМ количество результатов (например, до 1-2)
    if vector_store:
        docs = vector_store.similarity_search(query, k=1)  # Было k=2
        context = "\n".join([doc.page_content[:500] for doc in docs])  # Ограничиваем до 500 символов

    # Формируем итоговый запрос
    query = f"{instruction[:500]}\n\nКонтекст:\n{knowledge[:1000]}\n{context}\n\nВопрос: {query}"  # Ограничиваем текст

    client = openai.Client(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Ты - эксперт, используй контекст при ответе."},
                      {"role": "user", "content": query}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ошибка при запросе к OpenAI: {e}"

# Запуск бота
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
