import os
import logging
import openai
import asyncio
import shutil
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import CommandStart
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Настройки бота
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Отсутствует OPENAI_API_KEY. Добавьте его в переменные Railway.")

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Логирование
logging.basicConfig(level=logging.INFO)

# Пути к файлам
INDEX_FOLDER = "faiss_index"
INDEX_ZIP = "faiss_index.zip"
KNOWLEDGE_FILE = "knowledge.txt"

# Функция разбиения текста на чанки
def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100):
    """
    Разбивает текст на чанки заданного размера с перекрытием.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

# Проверяем, есть ли база знаний и FAISS-индекс
if os.path.exists(INDEX_ZIP):
    shutil.unpack_archive(INDEX_ZIP, INDEX_FOLDER)
    vector_store = FAISS.load_local(INDEX_FOLDER, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
else:
    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            knowledge_text = f.read()

        # Разбиваем на чанки и создаем FAISS
        docs = split_text_into_chunks(knowledge_text, chunk_size=500, overlap=100)
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
        vector_store.save_local(INDEX_FOLDER)
    else:
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
    context = ""

    if vector_store:
        docs = vector_store.similarity_search(query, k=2)  # Ищем 2 самых релевантных чанка
        context = "\n".join([doc.page_content for doc in docs])

    prompt = f"Контекст:\n{context}\n\nВопрос: {query}"

    client = openai.Client(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Ты - эксперт, используй контекст при ответе."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ошибка при запросе к OpenAI: {e}"

# Запуск бота
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
