import os
import logging
import openai
import asyncio
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
KNOWLEDGE_FILE = "knowledge.txt"
INSTRUCTION_FILE = "instruction.txt"

# Загрузка инструкции
if os.path.exists(INSTRUCTION_FILE):
    with open(INSTRUCTION_FILE, "r", encoding="utf-8") as f:
        system_instruction = f.read().strip()
    print("✅ Инструкция загружена.")
else:
    system_instruction = "Ты – эксперт, используй только указанный контекст."
    print("⚠ Файл instruction.txt не найден! Используется стандартная инструкция.")

# Функция разбиения текста на чанки
def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

# Загрузка базы знаний и создание FAISS-индекса на лету
if os.path.exists(KNOWLEDGE_FILE):
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        knowledge_text = f.read()
    print(f"🔍 База знаний загружена. Размер: {len(knowledge_text)} символов.")

    # Разбиваем на чанки и создаем FAISS
    docs = split_text_into_chunks(knowledge_text, chunk_size=500, overlap=100)
    if docs:
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
        print("✅ FAISS-индекс успешно создан в памяти!")
    else:
        print("❌ Ошибка: база знаний пуста или не разбита на чанки!")
        vector_store = None
else:
    print("❌ Файл knowledge.txt не найден!")
    vector_store = None

# Функция обработки команды /start
@dp.message(CommandStart())
async def start_handler(message: Message):
    await message.answer("Привет! Чем могу помочь!")

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
        docs = vector_store.similarity_search(query, k=2)
        if docs:
            context = "\n".join([doc.page_content for doc in docs])
            print(f"🔍 Найден контекст: {context}")

    if not context:
        return (
    "Извините, но я консультирую только по ТЕХНИЧЕСКому РЕГЛАМЕНТу ТАМОЖЕННОГО СОЮЗА "
    "'О БЕЗОПАСНОСТИ ЖЕЛЕЗНОДОРОЖНОГО ПОДВИЖНОГО СОСТАВА'. "
    "Вам стоит обратиться к профильным специалистам или официальным источникам информации."
)

    prompt = f"Контекст из базы:\n{context}\n\nВопрос: {query}"

    client = openai.Client(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_instruction},  # Инструкция загружается из файла
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ошибка при запросе к OpenAI: {e}"

# Запуск бота
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
