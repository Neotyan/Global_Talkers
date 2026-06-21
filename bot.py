import os
import re
import math
import json
import random
import asyncio
from io import BytesIO
from os import getenv
from datetime import datetime, timezone
from dotenv import load_dotenv
from aiohttp import web
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State

# === MongoDB ===
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReturnDocument

load_dotenv()
TOKEN = getenv("BOT_TOKEN")
ADMIN_CHAT_ID = getenv("ADMIN_CHAT_ID")

# ID админов (безлимитный ИИ). В .env: ADMIN_IDS=123456789,987654321
ADMIN_IDS = {int(p.strip()) for p in getenv("ADMIN_IDS", "").split(",") if p.strip().isdigit()}


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

# В .env переменная может называться MONGO_URL или DB_URL — поддерживаем оба варианта.
MONGO_URL = getenv("MONGO_URL") or getenv("DB_URL")

# Ключ для нейросети Gemini (если не задан — ИИ-помощник просто отключится без ошибок).
GEMINI_API_KEY = getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.5-flash"        # генерация ответов (лёгкая и быстрая)
EMBED_MODEL = "gemini-embedding-001"     # эмбеддинги для семантического поиска по кэшу
# Порог косинусной близости: выше — считаем вопросы одинаковыми по смыслу (0..1)
SIMILARITY_THRESHOLD = 0.86

# Асинхронный клиент google-genai (создаём только если есть ключ)
genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# === ПОДКЛЮЧЕНИЕ К БАЗЕ ДАННЫХ ===
mongo_client = AsyncIOMotorClient(MONGO_URL)
db = mongo_client["global_talkers"]
counters = db["counters"]      # коллекция-счётчик для автоинкремента
students = db["students"]      # анкеты учеников
teachers = db["teachers"]      # анкеты волонтёров (преподавателей)
ai_cache = db["ai_cache"]      # кэш ответов нейросети (частые вопросы)
ai_limits = db["ai_limits"]    # суточные лимиты ИИ-запросов на пользователя

# --- Защита / лимиты ИИ ---
DAILY_AI_LIMIT = 10            # сколько ИИ-вопросов в день на одного человека
VOICE_MAX_SECONDS = 120        # максимум 2 минуты на голосовое/кружочек
RATE_LIMITED = "__RATE_LIMITED__"  # маркер: API превысил лимит (429)

OVERHEAT_MSG = (
    "Ой, я сегодня ответил на слишком много вопросов и немного перегрелся 🥵.\n"
    "Но мои базовые функции работают! Жми /resources для изучения материалов."
)
LIMIT_REACHED_MSG = (
    "Твои бесплатные ИИ-вопросы на сегодня закончились! 🌙\n"
    "Пора сделать перерыв и закрепить теорию. Возвращайся завтра!\n\n"
    "А пока загляни в /resources и /grammar — там много полезного."
)

# Имя и id бота подтянем при старте (нужно для ИИ-помощника: упоминание и Reply на бота)
BOT_USERNAME = None
BOT_ID = None

# Карта статусов для красивого отображения
STATUS_LABELS = {
    "free": "🟢 Свободен (готов(а) к занятиям, пары сейчас нет)",
    "busy": "🔴 Занят (сейчас уже есть ученик/преподаватель)",
}

dp = Dispatcher()
router = Router()
dp.include_router(router)


# === АВТОИНКРЕМЕНТ НОМЕРОВ АНКЕТ ===
async def get_next_sequence(sequence_name: str) -> int:
    """Возвращает следующий номер (1, 2, 3...) для указанного счётчика."""
    result = await counters.find_one_and_update(
        {"_id": sequence_name},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return result["seq"]


# === КЭШ ВОПРОСОВ ===
def normalize_question(text: str) -> str:
    """Приводит вопрос к единому виду, чтобы ловить одинаковые формулировки."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)   # убираем пунктуацию (?, !, точки и т.д.)
    text = re.sub(r"\s+", " ", text)      # схлопываем повторные пробелы
    return text


async def reply_markdown(message: Message, text: str):
    """Отправляет ответ в Markdown, а при поломке разметки — обычным текстом."""
    try:
        await message.reply(text, parse_mode="Markdown")
    except Exception:
        await message.reply(text)


async def get_embedding(text: str) -> list | None:
    """Получает вектор-эмбеддинг вопроса через Gemini (для семантического поиска)."""
    if not genai_client:
        return None
    try:
        response = await genai_client.aio.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"Ошибка получения эмбеддинга: {e}")
        return None


def cosine_similarity(a: list, b: list) -> float:
    """Косинусная близость двух векторов (1.0 — идентичны по смыслу, 0 — нет связи)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def find_similar_cached(question_embedding: list):
    """Ищет в кэше самый близкий по смыслу вопрос. Возвращает документ или None."""
    best_doc, best_score = None, 0.0
    cursor = ai_cache.find(
        {"embedding": {"$exists": True}},
        {"answer": 1, "embedding": 1, "question": 1},
    )
    async for doc in cursor:
        score = cosine_similarity(question_embedding, doc.get("embedding", []))
        if score > best_score:
            best_doc, best_score = doc, score

    if best_doc and best_score >= SIMILARITY_THRESHOLD:
        print(f"Кэш-хит по смыслу (score={best_score:.3f}): {best_doc.get('question')!r}")
        return best_doc
    return None


async def consume_ai_quota(tg_id: int) -> bool:
    """Списывает 1 ИИ-вопрос из суточного лимита. True — можно, False — лимит исчерпан.

    У админов — безлимит. Счётчик сбрасывается каждый день (по UTC).
    """
    if is_admin(tg_id):
        return True

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    doc = await ai_limits.find_one({"_id": tg_id})

    # Новый день или первый запрос — обнуляем и ставим 1
    if not doc or doc.get("date") != today:
        await ai_limits.update_one(
            {"_id": tg_id},
            {"$set": {"date": today, "count": 1}},
            upsert=True,
        )
        return True

    # Лимит уже выбран
    if doc.get("count", 0) >= DAILY_AI_LIMIT:
        return False

    # Списываем ещё один
    await ai_limits.update_one({"_id": tg_id}, {"$inc": {"count": 1}})
    return True


async def get_ai_remaining(tg_id: int):
    """Сколько ИИ-вопросов осталось сегодня. None — безлимит (админ)."""
    if is_admin(tg_id):
        return None
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    doc = await ai_limits.find_one({"_id": tg_id})
    if not doc or doc.get("date") != today:
        return DAILY_AI_LIMIT
    return max(0, DAILY_AI_LIMIT - doc.get("count", 0))


def ai_quota_line(remaining) -> str:
    """Готовая строка про остаток ИИ-вопросов для /status."""
    if remaining is None:
        return "🤖 ИИ-вопросы: ♾ безлимит (админ)"
    return f"🤖 ИИ-вопросов сегодня: {remaining}/{DAILY_AI_LIMIT}"


SYSTEM_PROMPT = (
    "Ты — лаконичный репетитор по английскому в проекте Global Talkers. "
    "Отвечай максимально коротко и строго по делу, не более 5 предложений. "
    "Никаких длинных вступлений и воды — сразу суть и один короткий пример. "
    "Используй только базовый Markdown. Обязательно закрывай парные теги "
    "(например, *жирный* или _курсив_). Избегай сложных и вложенных списков. "
    "Отвечай на языке вопроса."
)


# === ОБРАЩЕНИЕ К НЕЙРОСЕТИ (GEMINI) ===
async def ask_gemini(question: str) -> str | None:
    """Отправляет вопрос в Gemini и возвращает текстовый ответ (или None при ошибке)."""
    if not genai_client:
        return None
    try:
        response = await genai_client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=question,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        return response.text
    except genai_errors.APIError as e:
        # 429 / RESOURCE_EXHAUSTED — превышен лимит запросов к API
        if e.code == 429 or e.status == "RESOURCE_EXHAUSTED":
            print(f"Gemini rate limit (429): {e}")
            return RATE_LIMITED
        print(f"Ошибка API Gemini: {e}")
        return None
    except Exception as e:
        print(f"Ошибка обращения к Gemini: {e}")
        return None


# === СОСТОЯНИЯ (ШАГИ АНКЕТЫ) ===
class TeacherForm(StatesGroup):
    name = State()
    age = State()
    level = State()
    proof = State()
    experience = State()
    motivation = State()
    hours = State()
    schedule = State()   # точное удобное время (например, с 14:00 до 15:00)
    confirm = State()    # Шаг проверки

class StudentForm(StatesGroup):
    name = State()
    age = State()
    level = State()
    goal = State()
    hours = State()
    time = State()
    schedule = State()   # точное удобное время (например, с 14:00 до 15:00)
    confirm = State()    # Шаг проверки


# Список всех команд бота (показываем в /start и /help)
def commands_help() -> str:
    mention = f"@{BOT_USERNAME}" if BOT_USERNAME else "@бота"
    return (
        "🤖 <b>Что я умею:</b>\n\n"
        "/start — выбрать роль и заполнить анкету\n"
        "/status — посмотреть и сменить свой статус (свободен/занят)\n"
        "/resources — материалы для самостоятельного изучения 📚\n"
        "/grammar — карманный справочник по грамматике 📖\n"
        "/topic — случайная тема для дискуссии (в группе) 🗣\n"
        "/quiz — викторина-опрос для учеников (в группе) ❓\n"
        "/help — показать это сообщение\n\n"
        "💬 <b>ИИ-помощник:</b> в личке просто напиши вопрос по английскому. "
        f"В группе — упомяни меня <b>{mention}</b>, начни сообщение со слова «Бот,» или «?», "
        "либо ответь (Reply) на моё сообщение.\n"
        "🎤 <b>Голосовые:</b> запиши голосовое или кружочек на английском — я расшифрую и дам фидбэк по речи!"
    )


# === СТАРТ И ВЫБОР РОЛИ ===
@router.message(CommandStart())
async def start_cmd(message: Message, state: FSMContext):
    await state.clear()

    text = (
        "Привет! Ты в официальном боте Global Talkers.\n\n"
        "Мы волонтерская языковая организация. Наша цель проста: помочь людям выучить английский язык абсолютно бесплатно.\n\n"
        "Мы объединяем тех, кто уже круто говорит по-английски и хочет попробовать себя в роли преподавателя, с теми, кому нужна помощь в изучении языка.\n\n"
        "Всё держится на энтузиазме, желании помогать и взаимном уважении. Если тебе близок такой подход, выбирай свою роль в меню и добро пожаловать в команду!\n\n"
        + commands_help()
    )

    markup = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Хочу преподавать (Волонтёр)", callback_data="role_teacher")],
        [InlineKeyboardButton(text="Хочу учиться (Ученик)", callback_data="role_student")],
        [InlineKeyboardButton(text="📚 Материалы для изучения", callback_data="res_menu")]
    ])

    try:
        await message.answer(text, reply_markup=markup, parse_mode="HTML")
    except Exception as e:
        print(f"Не удалось отправить приветствие пользователю {message.from_user.id}: {e}")

# === СПРАВКА ПО КОМАНДАМ ===
@router.message(Command("help"))
async def help_cmd(message: Message):
    await message.answer(commands_help(), parse_mode="HTML")


# === ОБРАБОТКА КНОПОК МЕНЮ ===
@router.callback_query(F.data == "role_teacher")
async def start_teacher_form(call: CallbackQuery, state: FSMContext):
    await state.update_data(username=call.from_user.username, tg_id=call.from_user.id)
    text = (
        "Круто! Давай заполним небольшую анкету преподавателя, чтобы мы могли с тобой познакомиться.\n\n"
        "1. Как к тебе обращаться? (Твое реальное имя или супергеройское прозвище)"
    )
    await call.message.answer(text)
    await state.set_state(TeacherForm.name)
    await call.answer()

@router.callback_query(F.data == "role_student")
async def start_student_form(call: CallbackQuery, state: FSMContext):
    await state.update_data(username=call.from_user.username, tg_id=call.from_user.id)
    text = (
        "Отличный выбор! Заполни анкету ниже, и мы подберем тебе преподавателя.\n\n"
        "1. Как к тебе обращаться? (Твое имя)"
    )
    await call.message.answer(text)
    await state.set_state(StudentForm.name)
    await call.answer()


# ================= ВЕТКА ВОЛОНТЁРА =================
@router.message(TeacherForm.name)
async def t_age(message: Message, state: FSMContext):
    await state.update_data(name=message.text)
    await message.answer("2. Сколько тебе лет?")
    await state.set_state(TeacherForm.age)

@router.message(TeacherForm.age)
async def t_level(message: Message, state: FSMContext):
    if not message.text.isdigit():
        await message.answer("Ошибка! Пожалуйста, напиши свой возраст просто цифрами (например: 15).")
        return

    await state.update_data(age=message.text)
    await message.answer("3. Какой у тебя сейчас уровень английского? (Нам нужны ребята от твердого B2 и выше)")
    await state.set_state(TeacherForm.level)

@router.message(TeacherForm.level)
async def t_proof(message: Message, state: FSMContext):
    await state.update_data(level=message.text)
    await message.answer("4. Чем можешь подтвердить свой левел? (Сертификат IELTS/TOEFL, справка с курсов, результаты онлайн-теста EF SET. Если ничего нет на руках не страшно, просто напиши, откуда язык).")
    await state.set_state(TeacherForm.proof)

@router.message(TeacherForm.proof)
async def t_exp(message: Message, state: FSMContext):
    await state.update_data(proof=message.text)
    await message.answer("5. Был ли у тебя опыт преподавания? (Даже если просто подтягивал младшего брата перед контрольной или объяснял тему одноклассникам на пальцах смело пиши).")
    await state.set_state(TeacherForm.experience)

@router.message(TeacherForm.experience)
async def t_motivation(message: Message, state: FSMContext):
    await state.update_data(experience=message.text)
    await message.answer("6. Почему ты хочешь стать волонтером в этом проекте? (Честно: скучаешь на каникулах, хочешь наработать часы волонтерства для универа, любишь помогать людям? Любой ответ принимается).")
    await state.set_state(TeacherForm.motivation)

@router.message(TeacherForm.motivation)
async def t_hours(message: Message, state: FSMContext):
    await state.update_data(motivation=message.text)
    await message.answer("7. Сколько часов в неделю ты сможешь стабильно уделять ученикам? (Лучше написать меньше, но честно, чтобы мы могли нормально составить график).")
    await state.set_state(TeacherForm.hours)

@router.message(TeacherForm.hours)
async def t_schedule(message: Message, state: FSMContext):
    await state.update_data(hours=message.text)
    await message.answer("8. В какое конкретное время тебе удобно проводить занятия? (Например: будни с 18:00 до 20:00, или выходные с 14:00 до 15:00)")
    await state.set_state(TeacherForm.schedule)

# Показываем превью анкеты Волонтёра
@router.message(TeacherForm.schedule)
async def t_preview(message: Message, state: FSMContext):
    await state.update_data(schedule=message.text)
    data = await state.get_data()

    text = (
        f"Твоя анкета готова! Проверь, всё ли верно:\n\n"
        f"<b>Имя:</b> {data['name']}\n"
        f"<b>Возраст:</b> {data['age']}\n"
        f"<b>Уровень:</b> {data['level']}\n"
        f"<b>Пруфы:</b> {data['proof']}\n"
        f"<b>Опыт:</b> {data['experience']}\n"
        f"<b>Мотивация:</b> {data['motivation']}\n"
        f"<b>Часы в неделю:</b> {data['hours']}\n"
        f"<b>Удобное время:</b> {data['schedule']}"
    )

    markup = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Всё верно, отправить", callback_data="t_send")],
        [InlineKeyboardButton(text="🔄 Заполнить заново", callback_data="t_edit")]
    ])

    await message.answer(text, reply_markup=markup, parse_mode="HTML")
    await state.set_state(TeacherForm.confirm)

# Кнопки финальной проверки Волонтёра
@router.callback_query(TeacherForm.confirm, F.data == "t_send")
async def t_send_final(call: CallbackQuery, state: FSMContext, bot: Bot):
    await call.message.edit_reply_markup(reply_markup=None)  # Прячем кнопки, чтобы не нажал дважды
    data = await state.get_data()
    username = f"@{data['username']}" if data.get('username') else "Скрыт"

    # 1. Генерируем номер анкеты и сохраняем в базу ПЕРЕД отправкой админам
    teacher_id = await get_next_sequence("teacher_id")
    doc = {
        "teacher_id": teacher_id,
        "tg_id": data.get("tg_id"),
        "username": data.get("username"),
        "name": data["name"],
        "age": data["age"],
        "level": data["level"],
        "proof": data["proof"],
        "experience": data["experience"],
        "motivation": data["motivation"],
        "hours": data["hours"],
        "schedule": data["schedule"],
        "status": "free",  # по умолчанию свободен
    }
    await teachers.insert_one(doc)

    text = (
        f"🎓 <b>Анкета Преподавателя #{teacher_id} | Global Talkers</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"👤 <b>Имя:</b> {data['name']}\n"
        f"🔗 <b>Ник:</b> {username}\n"
        f"🎂 <b>Возраст:</b> {data['age']}\n"
        f"📊 <b>Уровень:</b> {data['level']}\n"
        f"📜 <b>Пруфы:</b> {data['proof']}\n"
        f"🧑‍🏫 <b>Опыт:</b> {data['experience']}\n"
        f"🔥 <b>Мотивация:</b> {data['motivation']}\n"
        f"⏳ <b>Часы в неделю:</b> {data['hours']}\n"
        f"🕒 <b>Удобное время:</b> {data['schedule']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{STATUS_LABELS['free']}"
    )

    # 2. Отправляем в админку
    sent_text = await bot.send_message(
        chat_id=ADMIN_CHAT_ID,
        text=text,
        parse_mode="HTML"
    )
    await bot.pin_chat_message(chat_id=ADMIN_CHAT_ID, message_id=sent_text.message_id, disable_notification=True)

    await call.message.answer(
        f"Супер! Твоя анкета №{teacher_id} отправлена админам. "
        "Обязательно проверь, чтобы у тебя были открыты личные сообщения, иначе мы не сможем тебе написать.\n\n"
        "💡 Команда /status — посмотреть и сменить свой статус (свободен/занят). Скоро свяжемся!"
    )
    await state.clear()

@router.callback_query(TeacherForm.confirm, F.data == "t_edit")
async def t_edit_final(call: CallbackQuery, state: FSMContext):
    await call.message.edit_reply_markup(reply_markup=None)
    await call.message.answer("Без проблем, давай начнем сначала.\n\n1. Как к тебе обращаться?")
    await state.set_state(TeacherForm.name)


# ================= ВЕТКА УЧЕНИКА =================
@router.message(StudentForm.name)
async def s_age(message: Message, state: FSMContext):
    await state.update_data(name=message.text)
    await message.answer("2. Сколько тебе лет?")
    await state.set_state(StudentForm.age)

@router.message(StudentForm.age)
async def s_level(message: Message, state: FSMContext):
    if not message.text.isdigit():
        await message.answer("Ошибка! Пожалуйста, напиши свой возраст просто цифрами (например: 15).")
        return

    await state.update_data(age=message.text)
    await message.answer("3. Как ты оцениваешь свой английский на данный момент? (Например: начинаю с нуля, знаю только базу со школы, могу читать со словарем, немного понимаю на слух).")
    await state.set_state(StudentForm.level)

@router.message(StudentForm.level)
async def s_goal(message: Message, state: FSMContext):
    await state.update_data(level=message.text)
    await message.answer("4. Какая у тебя главная цель? (Для чего тебе английский: для школьных экзаменов, понимать видео в оригинале, или что-то другое. Напиши кратко, чтобы мы понимали твой настрой).")
    await state.set_state(StudentForm.goal)

@router.message(StudentForm.goal)
async def s_hours(message: Message, state: FSMContext):
    await state.update_data(goal=message.text)
    await message.answer("5. Сколько часов в неделю ты готов стабильно уделять занятиям? (Нам важно, чтобы ты не забросил учебу через пару дней, поэтому оценивай свои силы реально).")
    await state.set_state(StudentForm.hours)

@router.message(StudentForm.hours)
async def s_time(message: Message, state: FSMContext):
    await state.update_data(hours=message.text)
    await message.answer("6. В какое время суток тебе удобнее всего заниматься? (Утро, день или вечер)")
    await state.set_state(StudentForm.time)

@router.message(StudentForm.time)
async def s_schedule(message: Message, state: FSMContext):
    await state.update_data(time=message.text)
    await message.answer("7. Назови конкретное удобное время для занятий. (Например: будни с 14:00 до 15:00, или выходные после 18:00)")
    await state.set_state(StudentForm.schedule)

# Показываем превью анкеты Ученика
@router.message(StudentForm.schedule)
async def s_preview(message: Message, state: FSMContext):
    await state.update_data(schedule=message.text)
    data = await state.get_data()

    text = (
        f"Твоя анкета готова! Проверь, всё ли верно:\n\n"
        f"<b>Имя:</b> {data['name']}\n"
        f"<b>Возраст:</b> {data['age']}\n"
        f"<b>Уровень:</b> {data['level']}\n"
        f"<b>Цель:</b> {data['goal']}\n"
        f"<b>Часы в неделю:</b> {data['hours']}\n"
        f"<b>Время суток:</b> {data['time']}\n"
        f"<b>Удобное время:</b> {data['schedule']}"
    )

    markup = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Всё верно, отправить", callback_data="s_send")],
        [InlineKeyboardButton(text="🔄 Заполнить заново", callback_data="s_edit")]
    ])

    await message.answer(text, reply_markup=markup, parse_mode="HTML")
    await state.set_state(StudentForm.confirm)

# Кнопки финальной проверки Ученика
@router.callback_query(StudentForm.confirm, F.data == "s_send")
async def s_send_final(call: CallbackQuery, state: FSMContext, bot: Bot):
    await call.message.edit_reply_markup(reply_markup=None)
    data = await state.get_data()
    username = f"@{data['username']}" if data.get('username') else "Скрыт"

    # 1. Генерируем номер анкеты и сохраняем в базу ПЕРЕД отправкой админам
    student_id = await get_next_sequence("student_id")
    doc = {
        "student_id": student_id,
        "tg_id": data.get("tg_id"),
        "username": data.get("username"),
        "name": data["name"],
        "age": data["age"],
        "level": data["level"],
        "goal": data["goal"],
        "hours": data["hours"],
        "time": data["time"],
        "schedule": data["schedule"],
        "status": "free",  # по умолчанию свободен
    }
    await students.insert_one(doc)

    text = (
        f"📚 <b>Анкета Ученика #{student_id} | Global Talkers</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"👤 <b>Имя:</b> {data['name']}\n"
        f"🔗 <b>Ник:</b> {username}\n"
        f"🎂 <b>Возраст:</b> {data['age']}\n"
        f"📊 <b>Уровень:</b> {data['level']}\n"
        f"🎯 <b>Цель:</b> {data['goal']}\n"
        f"⏳ <b>Часы в неделю:</b> {data['hours']}\n"
        f"🌗 <b>Время суток:</b> {data['time']}\n"
        f"🕒 <b>Удобное время:</b> {data['schedule']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{STATUS_LABELS['free']}"
    )

    await bot.send_message(chat_id=ADMIN_CHAT_ID, text=text, parse_mode="HTML")
    await call.message.answer(
        f"Готово! Твоя анкета №{student_id} отправлена. "
        "Обязательно проверь, чтобы у тебя были открыты личные сообщения, иначе мы не сможем тебе написать.\n\n"
        "💡 Команда /status — посмотреть и сменить свой статус (свободен/занят). Скоро свяжемся!"
    )
    await state.clear()

@router.callback_query(StudentForm.confirm, F.data == "s_edit")
async def s_edit_final(call: CallbackQuery, state: FSMContext):
    await call.message.edit_reply_markup(reply_markup=None)
    await call.message.answer("Без проблем, давай начнем сначала.\n\n1. Как к тебе обращаться?")
    await state.set_state(StudentForm.name)


# ================= СТАТУС (свободен / занят) =================
async def find_user_record(tg_id: int):
    """Ищет запись пользователя по tg_id в обеих коллекциях. Возвращает (collection, doc)."""
    doc = await students.find_one({"tg_id": tg_id})
    if doc:
        return students, doc
    doc = await teachers.find_one({"tg_id": tg_id})
    if doc:
        return teachers, doc
    return None, None


def status_keyboard(current_status: str) -> InlineKeyboardMarkup:
    # Кнопка предлагает переключиться на противоположный статус
    if current_status == "free":
        btn = InlineKeyboardButton(text="🔴 Стать занятым", callback_data="set_status_busy")
    else:
        btn = InlineKeyboardButton(text="🟢 Стать свободным", callback_data="set_status_free")
    return InlineKeyboardMarkup(inline_keyboard=[[btn]])

@router.message(Command("status"))
async def status_cmd(message: Message):
    quota_line = ai_quota_line(await get_ai_remaining(message.from_user.id))

    collection, doc = await find_user_record(message.from_user.id)
    if not doc:
        await message.answer(
            "Я не нашёл твою анкету 🤔 Сначала заполни её через /start.\n\n" + quota_line
        )
        return

    current = doc.get("status", "free")
    await message.answer(
        f"Твой текущий статус:\n\n{STATUS_LABELS.get(current, current)}\n\n"
        "Свободен = сейчас нет ученика/преподавателя, готов(а) к новой паре.\n\n"
        f"{quota_line}",
        reply_markup=status_keyboard(current),
    )

@router.callback_query(F.data.startswith("set_status_"))
async def set_status(call: CallbackQuery):
    new_status = call.data.replace("set_status_", "")  # "free" или "busy"
    collection, doc = await find_user_record(call.from_user.id)
    if not doc:
        await call.answer("Анкета не найдена. Заполни её через /start.", show_alert=True)
        return

    await collection.update_one({"_id": doc["_id"]}, {"$set": {"status": new_status}})

    quota_line = ai_quota_line(await get_ai_remaining(call.from_user.id))
    await call.message.edit_text(
        f"Готово! Твой новый статус:\n\n{STATUS_LABELS.get(new_status, new_status)}\n\n{quota_line}",
        reply_markup=status_keyboard(new_status),
    )
    await call.answer("Статус обновлён ✅")


# ================= БАЗА ЗНАНИЙ / МАТЕРИАЛЫ (/resources) =================
# Категория -> (заголовок, [(название, ссылка, краткое описание), ...])
RESOURCES = {
    "yt": (
        "📺 YouTube-каналы",
        [
            ("BBC Learning English", "https://www.youtube.com/@bbclearningenglish", "Короткие уроки на любые темы"),
            ("English with Lucy", "https://www.youtube.com/@EnglishwithLucy", "Произношение и британский английский"),
            ("Linguamarina", "https://www.youtube.com/@linguamarina", "Лайфхаки и разговорный английский"),
        ],
    ),
    "pod": (
        "🎧 Подкасты",
        [
            ("The English We Speak (BBC)", "https://www.bbc.co.uk/learningenglish/features/the-english-we-speak", "Идиомы и фразы за 3 минуты"),
            ("6 Minute English (BBC)", "https://www.bbc.co.uk/learningenglish/features/6-minute-english", "Короткие диалоги на каждый день"),
            ("Luke's English Podcast", "https://teacherluke.co.uk/", "Живой английский с носителем"),
        ],
    ),
    "gram": (
        "📚 Грамматика",
        [
            ("EnglishGrammar.org", "https://www.englishgrammar.org/", "Правила и упражнения с ответами"),
            ("Perfect English Grammar", "https://www.perfect-english-grammar.com/", "Простые объяснения времён"),
            ("Grammarly Handbook", "https://www.grammarly.com/blog/category/handbook/", "Разбор частых ошибок"),
        ],
    ),
    "speak": (
        "🗣 Спикинг",
        [
            ("Tandem", "https://www.tandem.net/", "Языковой обмен с носителями"),
            ("Cambly", "https://www.cambly.com/", "Разговор с репетиторами онлайн"),
            ("ELSA Speak", "https://elsaspeak.com/", "Тренажёр произношения с ИИ"),
        ],
    ),
}

# Полезные ролики для кнопки «Случайное видео»
RANDOM_VIDEOS = [
    ("English with Lucy: 75 фраз для продвинутого английского", "https://www.youtube.com/watch?v=4jt5kFM3MJY"),
    ("Learn English with TV Series: 10 сериалов для изучения языка", "https://www.youtube.com/watch?v=4K9zbx-W8U4"),
    ("Learn English with TV Series: разбор лексики по мультфильму Elio", "https://www.youtube.com/watch?v=GXKORIHWjX0"),
    ("BBC 6 Minute English: как перестать бояться говорить", "https://www.youtube.com/watch?v=YAsDeXcYyTg"),
    ("BBC Learning English: как справляться со стрессом и волнением", "https://www.youtube.com/watch?v=fLB70DZdqIE"),
]


def resources_menu_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📺 YouTube-каналы", callback_data="res_yt")],
        [InlineKeyboardButton(text="🎧 Подкасты", callback_data="res_pod")],
        [InlineKeyboardButton(text="📚 Грамматика", callback_data="res_gram")],
        [InlineKeyboardButton(text="🗣 Спикинг", callback_data="res_speak")],
        [InlineKeyboardButton(text="🎲 Случайное видео", callback_data="res_random")],
    ])


def resources_back_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⬅️ Назад к категориям", callback_data="res_menu")]
    ])


RESOURCES_INTRO = (
    "📚 <b>Материалы для самостоятельного изучения</b>\n\n"
    "Выбери категорию — пришлю топ-3 проверенных ресурса.\n"
    "Не знаешь, с чего начать? Жми «🎲 Случайное видео»."
)


@router.message(Command("resources"))
async def resources_cmd(message: Message):
    await message.answer(RESOURCES_INTRO, reply_markup=resources_menu_markup(), parse_mode="HTML")

@router.callback_query(F.data == "res_menu")
async def resources_menu(call: CallbackQuery):
    try:
        await call.message.edit_text(RESOURCES_INTRO, reply_markup=resources_menu_markup(), parse_mode="HTML")
    except Exception:
        # Если сообщение нельзя отредактировать (например, это было /start) — шлём новое
        await call.message.answer(RESOURCES_INTRO, reply_markup=resources_menu_markup(), parse_mode="HTML")
    await call.answer()

@router.callback_query(F.data.startswith("res_") & ~F.data.in_({"res_menu", "res_random"}))
async def resources_category(call: CallbackQuery):
    key = call.data.replace("res_", "")
    category = RESOURCES.get(key)
    if not category:
        await call.answer("Категория не найдена")
        return

    title, items = category
    lines = [f"<b>{title} — топ-3</b>\n"]
    for i, (name, url, desc) in enumerate(items, start=1):
        lines.append(f"{i}. <a href=\"{url}\">{name}</a>\n    <i>{desc}</i>")
    text = "\n".join(lines)

    await call.message.edit_text(
        text, reply_markup=resources_back_markup(), parse_mode="HTML", disable_web_page_preview=True
    )
    await call.answer()

@router.callback_query(F.data == "res_random")
async def resources_random(call: CallbackQuery):
    name, url = random.choice(RANDOM_VIDEOS)
    text = (
        "🎲 <b>Случайное видео на сегодня</b>\n\n"
        f"<a href=\"{url}\">{name}</a>\n\n"
        "Не понравилось? Жми ещё раз 👇"
    )
    markup = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🎲 Ещё видео", callback_data="res_random")],
        [InlineKeyboardButton(text="⬅️ Назад к категориям", callback_data="res_menu")],
    ])
    await call.message.edit_text(text, reply_markup=markup, parse_mode="HTML", disable_web_page_preview=False)
    await call.answer()


# ================= ИНСТРУМЕНТЫ ДЛЯ ПРЕПОДАВАТЕЛЕЙ =================
# Темы для дискуссии (/topic)
DISCUSSION_TOPICS = [
    "If you could time travel, where would you go and why?",
    "What superpower would you choose and how would you use it?",
    "Is it better to be a leader or a follower? Why?",
    "If you could have dinner with anyone in history, who would it be?",
    "What is one thing you would change about your city?",
    "Would you rather be famous or rich? Explain your choice.",
    "If money was not a problem, what job would you do?",
    "What is the best advice you have ever received?",
    "Should students be allowed to use phones at school?",
    "If you could live in any country, where would it be and why?",
    "What invention has changed the world the most?",
    "Is social media good or bad for friendships?",
    "What would your perfect weekend look like?",
    "If you could master any skill instantly, what would it be?",
    "What does happiness mean to you?",
]

# Банк викторин (фолбэк, если ИИ недоступен)
QUIZ_BANK = [
    {"question": "Choose the correct word: She ___ to music every evening.",
     "options": ["listens", "listen", "listening", "listened"], "correct": 0,
     "explanation": "Present Simple, 3-е лицо ед. ч. → +s."},
    {"question": "What is the past form of 'go'?",
     "options": ["goed", "went", "gone", "going"], "correct": 1,
     "explanation": "'go' — неправильный глагол: go-went-gone."},
    {"question": "Choose the correct article: I saw ___ elephant at the zoo.",
     "options": ["a", "an", "the", "—"], "correct": 1,
     "explanation": "Перед гласным звуком ставится 'an'."},
    {"question": "Pick the synonym of 'happy':",
     "options": ["sad", "glad", "angry", "tired"], "correct": 1,
     "explanation": "'glad' = рад/счастлив."},
    {"question": "Complete: They have lived here ___ 2010.",
     "options": ["for", "since", "from", "during"], "correct": 1,
     "explanation": "'since' + точка отсчёта во времени."},
]


GROUP_ONLY_MSG = (
    "👥 Эта команда работает только в групповом чате с учениками.\n"
    "Добавь меня в группу и вызови команду там."
)

def is_group_chat(message: Message) -> bool:
    return message.chat.type in ("group", "supergroup")


@router.message(Command("topic"))
async def topic_cmd(message: Message):
    if not is_group_chat(message):
        await message.answer(GROUP_ONLY_MSG)
        return
    topic = random.choice(DISCUSSION_TOPICS)
    text = (
        "🗣 <b>Тема для дискуссии</b>\n\n"
        f"💬 <i>{topic}</i>\n\n"
        "Отлично заходит для разогрева в начале урока!"
    )
    markup = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔄 Другая тема", callback_data="topic_more")]
    ])
    await message.answer(text, reply_markup=markup, parse_mode="HTML")

@router.callback_query(F.data == "topic_more")
async def topic_more(call: CallbackQuery):
    topic = random.choice(DISCUSSION_TOPICS)
    text = (
        "🗣 <b>Тема для дискуссии</b>\n\n"
        f"💬 <i>{topic}</i>\n\n"
        "Отлично заходит для разогрева в начале урока!"
    )
    markup = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔄 Другая тема", callback_data="topic_more")]
    ])
    await call.message.edit_text(text, reply_markup=markup, parse_mode="HTML")
    await call.answer()


async def generate_quiz() -> dict:
    """Генерирует викторину через Gemini, при ошибке берёт вопрос из готового банка."""
    if genai_client:
        prompt = (
            "Сгенерируй ОДНУ простую викторину по английскому языку (уровень A2-B1) "
            "для проверки словарного запаса или базовой грамматики. "
            "Верни СТРОГО JSON без markdown и пояснений в формате: "
            '{"question": "...", "options": ["..", "..", "..", ".."], "correct": 0, "explanation": "..."}. '
            "Вопрос и варианты — на английском. Ровно 4 варианта, один правильный. "
            "Поле correct — индекс правильного варианта (0-3). "
            "explanation — короткое пояснение на русском (до 150 символов)."
        )
        try:
            response = await genai_client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            data = json.loads(response.text)
            options = data.get("options")
            correct = data.get("correct")
            if (isinstance(options, list) and 2 <= len(options) <= 10
                    and isinstance(correct, int) and 0 <= correct < len(options)
                    and data.get("question")):
                return data
        except Exception as e:
            print(f"Ошибка генерации квиза, беру из банка: {e}")

    return random.choice(QUIZ_BANK)

@router.message(Command("quiz"))
async def quiz_cmd(message: Message, bot: Bot):
    if not is_group_chat(message):
        await message.answer(GROUP_ONLY_MSG)
        return
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    quiz = await generate_quiz()

    # Телеграм лимиты: вопрос ≤300, вариант ≤100, пояснение ≤200 символов
    question = quiz["question"][:300]
    options = [str(opt)[:100] for opt in quiz["options"]]
    explanation = (quiz.get("explanation") or "")[:200] or None

    await bot.send_poll(
        chat_id=message.chat.id,
        question=question,
        options=options,
        type="quiz",
        correct_option_id=int(quiz["correct"]),
        is_anonymous=False,
        explanation=explanation,
    )


# ================= КАРМАННАЯ ГРАММАТИКА (/grammar) =================
GRAMMAR = {
    "tenses": (
        "⏳ Времена (Tenses)",
        "<b>⏳ Основные времена</b>\n\n"
        "<b>Present Simple</b> — факты, привычки\n"
        "<code>I work / She works</code>\n"
        "<i>маркеры: always, usually, every day</i>\n\n"
        "<b>Present Continuous</b> — прямо сейчас\n"
        "<code>I am working</code>\n"
        "<i>маркеры: now, at the moment</i>\n\n"
        "<b>Present Perfect</b> — есть связь с настоящим / опыт\n"
        "<code>I have worked</code>\n"
        "<i>маркеры: already, just, ever, never, yet</i>\n\n"
        "<b>Past Simple</b> — законченное действие в прошлом\n"
        "<code>I worked</code>\n"
        "<i>маркеры: yesterday, ago, in 2010</i>\n\n"
        "<b>Future Simple</b> — решение/прогноз\n"
        "<code>I will work</code>\n"
        "<i>маркеры: tomorrow, next week</i>",
    ),
    "articles": (
        "🅰️ Артикли (a / an / the)",
        "<b>🅰️ Артикли</b>\n\n"
        "<b>a / an</b> — неопределённый, предмет «один из многих», впервые упомянут:\n"
        "<code>a cat</code> (перед согласным звуком),\n"
        "<code>an apple</code> (перед гласным звуком).\n\n"
        "<b>the</b> — определённый, конкретный/уже известный предмет, единственный в своём роде:\n"
        "<code>the sun, the book on the table</code>.\n\n"
        "<b>— (без артикля)</b> — с именами, языками, спортом, во множественном числе в общем смысле:\n"
        "<code>I like cats. He speaks English.</code>",
    ),
    "prepositions": (
        "🔗 Предлоги (in / on / at)",
        "<b>🔗 Предлоги места и времени</b>\n\n"
        "<b>Время:</b>\n"
        "<b>at</b> — точное время: <code>at 5 o'clock, at night</code>\n"
        "<b>on</b> — дни и даты: <code>on Monday, on July 4th</code>\n"
        "<b>in</b> — месяцы, годы, части суток: <code>in May, in 2020, in the morning</code>\n\n"
        "<b>Место:</b>\n"
        "<b>at</b> — точка: <code>at the door, at the bus stop</code>\n"
        "<b>on</b> — на поверхности: <code>on the table, on the wall</code>\n"
        "<b>in</b> — внутри: <code>in the box, in London</code>",
    ),
    "conditionals": (
        "❓ Условные предложения",
        "<b>❓ Conditionals</b>\n\n"
        "<b>Zero</b> — всегда истина (факты):\n"
        "<code>If you heat ice, it melts.</code>\n\n"
        "<b>First</b> — реальное будущее:\n"
        "<code>If it rains, I will stay home.</code>\n"
        "<i>(if + Present Simple, → will + V)</i>\n\n"
        "<b>Second</b> — нереальное настоящее/мечты:\n"
        "<code>If I had money, I would travel.</code>\n"
        "<i>(if + Past Simple, → would + V)</i>\n\n"
        "<b>Third</b> — нереальное прошлое (сожаление):\n"
        "<code>If I had studied, I would have passed.</code>\n"
        "<i>(if + Past Perfect, → would have + V3)</i>",
    ),
}

GRAMMAR_INTRO = (
    "📖 <b>Карманная грамматика</b>\n\n"
    "Быстро вспомнить правило прямо здесь и сейчас. Выбери тему 👇"
)


def grammar_menu_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⏳ Времена (Tenses)", callback_data="gram_tenses")],
        [InlineKeyboardButton(text="🅰️ Артикли (a / an / the)", callback_data="gram_articles")],
        [InlineKeyboardButton(text="🔗 Предлоги (in / on / at)", callback_data="gram_prepositions")],
        [InlineKeyboardButton(text="❓ Условные предложения", callback_data="gram_conditionals")],
    ])


def grammar_back_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⬅️ Назад к темам", callback_data="gram_menu")]
    ])


@router.message(Command("grammar"))
async def grammar_cmd(message: Message):
    await message.answer(GRAMMAR_INTRO, reply_markup=grammar_menu_markup(), parse_mode="HTML")

@router.callback_query(F.data == "gram_menu")
async def grammar_menu(call: CallbackQuery):
    try:
        await call.message.edit_text(GRAMMAR_INTRO, reply_markup=grammar_menu_markup(), parse_mode="HTML")
    except Exception:
        await call.message.answer(GRAMMAR_INTRO, reply_markup=grammar_menu_markup(), parse_mode="HTML")
    await call.answer()

@router.callback_query(F.data.startswith("gram_") & ~F.data.in_({"gram_menu"}))
async def grammar_topic(call: CallbackQuery):
    key = call.data.replace("gram_", "")
    topic = GRAMMAR.get(key)
    if not topic:
        await call.answer("Тема не найдена")
        return
    _, content = topic
    await call.message.edit_text(content, reply_markup=grammar_back_markup(), parse_mode="HTML")
    await call.answer()


# ================= ГОЛОСОВЫЕ СООБЩЕНИЯ (расшифровка + фидбэк) =================
async def transcribe_and_feedback(audio_bytes: bytes, mime_type: str) -> str | None:
    """Отдаёт аудио напрямую в Gemini: он расшифровывает речь и даёт фидбэк по английскому."""
    if not genai_client:
        return None
    prompt = (
        "Это голосовое сообщение ученика, который тренирует разговорный английский. "
        "1) Сначала расшифруй, что он сказал (транскрипция на английском). "
        "2) Затем дай дружелюбный короткий фидбэк на русском: что звучит хорошо, "
        "какие есть ошибки в грамматике/словах и как сказать естественнее. "
        "Если речь не на английском — мягко попроси записать на английском.\n\n"
        "Формат строго такой:\n"
        "🗣 *Ты сказал:* <транскрипция>\n\n"
        "✅ *Фидбэк:* <короткий разбор, не более 5 предложений>\n\n"
        "Используй только базовый Markdown, закрывай парные теги."
    )
    try:
        response = await genai_client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)],
        )
        return response.text
    except genai_errors.APIError as e:
        if e.code == 429 or e.status == "RESOURCE_EXHAUSTED":
            print(f"Gemini rate limit (429) на голосовом: {e}")
            return RATE_LIMITED
        print(f"Ошибка API Gemini (голос): {e}")
        return None
    except Exception as e:
        print(f"Ошибка обработки голосового: {e}")
        return None

@router.message(F.voice | F.video_note)
async def voice_handler(message: Message, state: FSMContext, bot: Bot):
    # Не вмешиваемся, если человек прямо сейчас заполняет анкету
    if await state.get_state() is not None:
        return

    if not genai_client:
        await message.reply("Голосовой разбор пока не настроен (нет ключа GEMINI_API_KEY).")
        return

    # voice — это голосовое (audio/ogg), video_note — кружочек (видео mp4 со звуком)
    if message.voice:
        file_id = message.voice.file_id
        duration = message.voice.duration or 0
        mime_type = "audio/ogg"
    else:
        file_id = message.video_note.file_id
        duration = message.video_note.duration or 0
        mime_type = "video/mp4"

    # Лимит длины — не дольше 2 минут
    if duration > VOICE_MAX_SECONDS:
        await message.reply(
            "Запись слишком длинная 🙏 Максимум 2 минуты.\n"
            "Раздели мысль на части и пришли покороче — так и разбор будет точнее."
        )
        return

    # Списываем ИИ-запрос из суточного лимита пользователя
    if not await consume_ai_quota(message.from_user.id):
        await message.reply(LIMIT_REACHED_MSG)
        return

    await bot.send_chat_action(chat_id=message.chat.id, action="typing")

    # Скачиваем файл в память и отправляем в Gemini
    buf = BytesIO()
    await bot.download(file_id, destination=buf)
    audio_bytes = buf.getvalue()

    result = await transcribe_and_feedback(audio_bytes, mime_type)
    if result == RATE_LIMITED:
        await message.reply(OVERHEAT_MSG)
    elif result:
        await reply_markdown(message, result)
    else:
        await message.reply("Не получилось разобрать запись 🙈 Попробуй записать ещё раз, чуть чётче.")


# ================= ИИ-ПОМОЩНИК (по упоминанию @бота) =================
@router.message(F.text)
async def ai_assistant(message: Message, state: FSMContext, bot: Bot):
    # Не вмешиваемся, если человек прямо сейчас заполняет анкету
    if await state.get_state() is not None:
        return

    text = message.text or ""
    if text.startswith("/"):  # команды обрабатывают свои хендлеры
        return

    question = None

    if message.chat.type == "private":
        # В личке — отвечаем на любой вопрос
        question = text.strip()
    else:
        # В ГРУППЕ отвечаем только если к нам явно обратились:
        # 1) упомянули @бота, 2) ответили (Reply) на наше сообщение,
        # 3) начали сообщение со слова «Бот,» / «Bot», 4) начали с «?»
        low = text.lower()
        mention = f"@{BOT_USERNAME}".lower() if BOT_USERNAME else None
        is_reply_to_bot = bool(
            message.reply_to_message
            and message.reply_to_message.from_user
            and message.reply_to_message.from_user.id == BOT_ID
        )

        if mention and mention in low:
            question = text.replace(f"@{BOT_USERNAME}", "").strip()
        elif is_reply_to_bot:
            question = text.strip()
        elif re.match(r"^\s*(бот|bot)\b", text, flags=re.IGNORECASE):
            question = re.sub(r"^\s*(бот|bot)[\s,:!.\-]*", "", text, flags=re.IGNORECASE).strip()
        elif text.lstrip().startswith("?"):
            question = text.lstrip().lstrip("?").strip()
        else:
            return  # к боту не обращались — молчим, не лезем в чужой разговор

    if not question:
        await message.reply(
            "Привет! Задай мне вопрос по английскому, например:\n"
            "«В чём разница между Present Perfect и Past Simple?»"
        )
        return

    # УРОВЕНЬ 1. Точное совпадение по нормализованному ключу — мгновенно, без API
    cache_key = normalize_question(question)
    cached = await ai_cache.find_one({"key": cache_key})
    if cached:
        await reply_markdown(message, cached["answer"])
        return

    if not GEMINI_API_KEY:
        await message.reply("ИИ-помощник пока не настроен (нет ключа GEMINI_API_KEY).")
        return

    # Списываем ИИ-вопрос из суточного лимита (точные кэш-хиты выше — бесплатные)
    if not await consume_ai_quota(message.from_user.id):
        await message.reply(LIMIT_REACHED_MSG)
        return

    await bot.send_chat_action(chat_id=message.chat.id, action="typing")

    # УРОВЕНЬ 2. Семантический поиск: разные формулировки одного вопроса ловим по смыслу
    question_embedding = await get_embedding(question)
    if question_embedding:
        similar = await find_similar_cached(question_embedding)
        if similar:
            await reply_markdown(message, similar["answer"])
            return

    # УРОВЕНЬ 3. Похожего нет — генерируем ответ нейросетью
    answer = await ask_gemini(question)
    if answer == RATE_LIMITED:
        await message.reply(OVERHEAT_MSG)
        return
    if answer:
        # Сохраняем ответ + эмбеддинг в кэш (upsert — не плодим дубли)
        doc = {"key": cache_key, "question": question, "answer": answer}
        if question_embedding:
            doc["embedding"] = question_embedding
        await ai_cache.update_one({"key": cache_key}, {"$set": doc}, upsert=True)
        await reply_markdown(message, answer)
    else:
        await message.reply("Упс, не получилось получить ответ от нейросети. Попробуй переформулировать вопрос чуть позже 🙏")


# ================= ВЕБ-СЕРВЕР (keep-alive для хостинга) =================
async def handle_ping(request):
    return web.Response(text="Бот работает, не спать!")

async def main():
    global BOT_USERNAME, BOT_ID

    if not ADMIN_CHAT_ID:
        print("ВНИМАНИЕ: Не найден ADMIN_CHAT_ID! Проверь файл .env")
        return

    if not MONGO_URL:
        print("ВНИМАНИЕ: Не найден MONGO_URL / DB_URL! Проверь файл .env")
        return

    # Индекс на ключ кэша: быстрый поиск и защита от дублей
    await ai_cache.create_index("key", unique=True)

    bot = Bot(token=TOKEN)

    # Узнаём username и id бота — нужно для ИИ-помощника (упоминание и Reply)
    me = await bot.get_me()
    BOT_USERNAME = me.username
    BOT_ID = me.id
    print(f"Бот Global Talkers запущен... (@{BOT_USERNAME})")

    # Регистрируем меню команд (кнопка «/» в Telegram)
    from aiogram.types import BotCommand
    await bot.set_my_commands([
        BotCommand(command="start", description="Начать / заполнить анкету"),
        BotCommand(command="status", description="Мой статус (свободен/занят)"),
        BotCommand(command="resources", description="Материалы для изучения"),
        BotCommand(command="grammar", description="Карманная грамматика"),
        BotCommand(command="topic", description="Тема для дискуссии (в группе)"),
        BotCommand(command="quiz", description="Викторина (в группе)"),
        BotCommand(command="help", description="Список команд"),
    ])

    # 1. Запускаем Телеграм-бота параллельным процессом
    asyncio.create_task(dp.start_polling(bot))

    # 2. Включаем веб-обманку (keep-alive)
    app = web.Application()
    app.router.add_get('/', handle_ping)

    runner = web.AppRunner(app)
    await runner.setup()

    port = int(os.environ.get('PORT', 10000))
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()

    print(f"Сервер-обманка запущен на порту {port}...")

    # 3. Держим программу включенной бесконечно
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
