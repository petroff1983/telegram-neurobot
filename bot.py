import os
import logging
import openai
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import CommandStart
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Настройки бота
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Устанавливаем ключ API OpenAI
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Логирование
logging.basicConfig(level=logging.INFO)

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

# Функция запроса к OpenAI
def ask_ai(query: str) -> str:
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
