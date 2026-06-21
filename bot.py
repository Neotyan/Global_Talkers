import os
import asyncio
from os import getenv
from dotenv import load_dotenv
from aiohttp import web
import aiohttp
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

# В .env переменная может называться MONGO_URL или DB_URL — поддерживаем оба варианта.
MONGO_URL = getenv("MONGO_URL") or getenv("DB_URL")

# Ключ для нейросети Gemini (если не задан — ИИ-помощник просто отключится без ошибок).
GEMINI_API_KEY = getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# === ПОДКЛЮЧЕНИЕ К БАЗЕ ДАННЫХ ===
mongo_client = AsyncIOMotorClient(MONGO_URL)
db = mongo_client["global_talkers"]
counters = db["counters"]      # коллекция-счётчик для автоинкремента
students = db["students"]      # анкеты учеников
teachers = db["teachers"]      # анкеты волонтёров (преподавателей)

# Имя бота подтянем при старте (нужно для ИИ-помощника по упоминанию @бота)
BOT_USERNAME = None

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


# === ОБРАЩЕНИЕ К НЕЙРОСЕТИ (GEMINI) ===
async def ask_gemini(question: str) -> str | None:
    """Отправляет вопрос в Gemini и возвращает текстовый ответ (или None при ошибке)."""
    if not GEMINI_API_KEY:
        return None

    system_prompt = (
        "Ты — дружелюбный помощник по английскому языку в волонтёрском проекте Global Talkers. "
        "Отвечай кратко, понятно и по делу, с примерами. Если вопрос не про английский — "
        "всё равно постарайся помочь простыми словами. Отвечай на языке вопроса."
    )
    payload = {
        "contents": [{"parts": [{"text": f"{system_prompt}\n\nВопрос: {question}"}]}]
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
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


# === СТАРТ И ВЫБОР РОЛИ ===
@router.message(CommandStart())
async def start_cmd(message: Message, state: FSMContext):
    await state.clear()

    text = (
        "Привет! Ты в официальном боте Global Talkers.\n\n"
        "Мы волонтерская языковая организация. Наша цель проста: помочь людям выучить английский язык абсолютно бесплатно.\n\n"
        "Мы объединяем тех, кто уже круто говорит по-английски и хочет попробовать себя в роли преподавателя, с теми, кому нужна помощь в изучении языка.\n\n"
        "Всё держится на энтузиазме, желании помогать и взаимном уважении. Если тебе близок такой подход, выбирай свою роль в меню и добро пожаловать в команду!\n\n"
        "💡 Уже заполнил анкету? Команда /status — посмотреть и сменить свой статус (свободен/занят)."
    )

    markup = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Хочу преподавать (Волонтёр)", callback_data="role_teacher")],
        [InlineKeyboardButton(text="Хочу учиться (Ученик)", callback_data="role_student")]
    ])

    try:
        await message.answer(text, reply_markup=markup)
    except Exception as e:
        print(f"Не удалось отправить приветствие пользователю {message.from_user.id}: {e}")


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
    collection, doc = await find_user_record(message.from_user.id)
    if not doc:
        await message.answer("Я не нашёл твою анкету 🤔 Сначала заполни её через /start.")
        return

    current = doc.get("status", "free")
    await message.answer(
        f"Твой текущий статус:\n\n{STATUS_LABELS.get(current, current)}\n\n"
        "Свободен = сейчас нет ученика/преподавателя, готов(а) к новой паре.",
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

    await call.message.edit_text(
        f"Готово! Твой новый статус:\n\n{STATUS_LABELS.get(new_status, new_status)}",
        reply_markup=status_keyboard(new_status),
    )
    await call.answer("Статус обновлён ✅")


# ================= ИИ-ПОМОЩНИК (по упоминанию @бота) =================
@router.message(F.text)
async def ai_assistant(message: Message, state: FSMContext, bot: Bot):
    # Не вмешиваемся, если человек прямо сейчас заполняет анкету
    if await state.get_state() is not None:
        return

    # Срабатываем только если упомянули бота (@username)
    if not BOT_USERNAME:
        return
    mention = f"@{BOT_USERNAME}".lower()
    if mention not in (message.text or "").lower():
        return

    # Вырезаем упоминание — остаётся сам вопрос
    question = message.text.replace(f"@{BOT_USERNAME}", "").strip()
    if not question:
        await message.reply("Привет! Задай мне вопрос по английскому, например:\n"
                            f"«@{BOT_USERNAME} в чём разница между Present Perfect и Past Simple?»")
        return

    if not GEMINI_API_KEY:
        await message.reply("ИИ-помощник пока не настроен (нет ключа GEMINI_API_KEY).")
        return

    # Показываем «печатает...», пока ждём ответ нейросети
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    answer = await ask_gemini(question)

    if answer:
        await message.reply(answer)
    else:
        await message.reply("Упс, не получилось получить ответ от нейросети. Попробуй переформулировать вопрос чуть позже 🙏")


# ================= ВЕБ-СЕРВЕР (keep-alive для хостинга) =================
async def handle_ping(request):
    return web.Response(text="Бот работает, не спать!")

async def main():
    global BOT_USERNAME

    if not ADMIN_CHAT_ID:
        print("ВНИМАНИЕ: Не найден ADMIN_CHAT_ID! Проверь файл .env")
        return

    if not MONGO_URL:
        print("ВНИМАНИЕ: Не найден MONGO_URL / DB_URL! Проверь файл .env")
        return

    bot = Bot(token=TOKEN)

    # Узнаём username бота — нужно для ИИ-помощника по упоминанию
    me = await bot.get_me()
    BOT_USERNAME = me.username
    print(f"Бот Global Talkers запущен... (@{BOT_USERNAME})")

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
