import asyncio
from os import getenv
from dotenv import load_dotenv
from aiohttp import web

from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State

load_dotenv()
TOKEN = getenv("BOT_TOKEN")
ADMIN_CHAT_ID = getenv("ADMIN_CHAT_ID")

dp = Dispatcher()
router = Router()
dp.include_router(router)

# === СОСТОЯНИЯ (ШАГИ АНКЕТЫ) ===
class TeacherForm(StatesGroup):
    name = State()
    age = State()
    level = State()
    proof = State()
    experience = State()
    motivation = State()
    hours = State()
    media = State()
    confirm = State() # Шаг проверки

class StudentForm(StatesGroup):
    name = State()
    age = State()
    level = State()
    goal = State()
    hours = State()
    time = State()
    confirm = State() # Шаг проверки

# === СТАРТ И ВЫБОР РОЛИ ===
@router.message(CommandStart())
async def start_cmd(message: Message, state: FSMContext):
    await state.clear()
    
    text = (
        "Привет! Ты в официальном боте Global Talkers.\n\n"
        "Мы волонтерская языковая организация. Наша цель проста: помочь людям выучить английский язык абсолютно бесплатно.\n\n"
        "Мы объединяем тех, кто уже круто говорит по-английски и хочет попробовать себя в роли преподавателя, с теми, кому нужна помощь в изучении языка.\n\n"
        "Всё держится на энтузиазме, желании помогать и взаимном уважении. Если тебе близок такой подход, выбирай свою роль в меню и добро пожаловать в команду!"
    )
    
    markup = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Хочу преподавать (Волонтёр)", callback_data="role_teacher")],
        [InlineKeyboardButton(text="Хочу учиться (Ученик)", callback_data="role_student")]
    ])
    
    await message.answer(text, reply_markup=markup)

# === ОБРАБОТКА КНОПОК МЕНЮ ===
@router.callback_query(F.data == "role_teacher")
async def start_teacher_form(call: CallbackQuery, state: FSMContext):
    await state.update_data(username=call.from_user.username)
    text = (
        "Круто! Давай заполним небольшую анкету преподавателя, чтобы мы могли с тобой познакомиться.\n\n"
        "1. Как к тебе обращаться? (Твое реальное имя или супергеройское прозвище)"
    )
    await call.message.answer(text)
    await state.set_state(TeacherForm.name)
    await call.answer()

@router.callback_query(F.data == "role_student")
async def start_student_form(call: CallbackQuery, state: FSMContext):
    await state.update_data(username=call.from_user.username)
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
async def t_media(message: Message, state: FSMContext):
    await state.update_data(hours=message.text)
    await message.answer("8. Покажи себя в деле: Запиши кружочек или голосовое сообщение на 1 минуту на английском и скинь его сюда. Расскажи немного о себе и своих увлечениях. (Так мы поймем, что ты хочешь участвовать, и будем знать твой разговорный уровень, несмотря на акцент).")
    await state.set_state(TeacherForm.media)

# Ловим кружочек и показываем превью анкеты
@router.message(TeacherForm.media, F.voice | F.video_note)
async def t_preview(message: Message, state: FSMContext):
    # Запоминаем ID сообщения с медиа
    await state.update_data(media_msg_id=message.message_id)
    data = await state.get_data()
    
    text = (
        f"Твоя анкета готова! Проверь, всё ли верно:\n\n"
        f"<b>Имя:</b> {data['name']}\n"
        f"<b>Возраст:</b> {data['age']}\n"
        f"<b>Уровень:</b> {data['level']}\n"
        f"<b>Пруфы:</b> {data['proof']}\n"
        f"<b>Опыт:</b> {data['experience']}\n"
        f"<b>Мотивация:</b> {data['motivation']}\n"
        f"<b>Часы в неделю:</b> {data['hours']}\n\n"
        f"<i>(Твое голосовое/видео тоже сохранено)</i>"
    )
    
    markup = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Всё верно, отправить", callback_data="t_send")],
        [InlineKeyboardButton(text="🔄 Заполнить заново", callback_data="t_edit")]
    ])
    
    await message.answer(text, reply_markup=markup, parse_mode="HTML")
    await state.set_state(TeacherForm.confirm)

@router.message(TeacherForm.media)
async def t_media_wrong(message: Message):
    await message.answer("Это не голосовое и не видеокружок. Пожалуйста, запиши аудио или видео!")

# Кнопки финальной проверки Волонтёра
@router.callback_query(TeacherForm.confirm, F.data == "t_send")
async def t_send_final(call: CallbackQuery, state: FSMContext, bot: Bot):
    await call.message.edit_reply_markup(reply_markup=None) # Прячем кнопки, чтобы не нажал дважды
    data = await state.get_data()
    username = f"@{data['username']}" if data.get('username') else "Скрыт"
    
    text = (
        f"🎓 <b>Анкета Преподавателя | Global Talkers</b>\n\n"
        f"<b>Имя:</b> {data['name']}\n"
        f"<b>Ник:</b> {username}\n"
        f"<b>Возраст:</b> {data['age']}\n"
        f"<b>Уровень:</b> {data['level']}\n"
        f"<b>Пруфы:</b> {data['proof']}\n"
        f"<b>Опыт:</b> {data['experience']}\n"
        f"<b>Мотивация:</b> {data['motivation']}\n"
        f"<b>Часы в неделю:</b> {data['hours']}"
    )
    
    # Отправляем в админку
    sent_media = await bot.copy_message(
        chat_id=ADMIN_CHAT_ID, 
        from_chat_id=call.message.chat.id, 
        message_id=data['media_msg_id']
    )
    sent_text = await bot.send_message(
        chat_id=ADMIN_CHAT_ID, 
        text=text, 
        parse_mode="HTML",
        reply_to_message_id=sent_media.message_id
    )
    await bot.pin_chat_message(chat_id=ADMIN_CHAT_ID, message_id=sent_text.message_id, disable_notification=True)
    
    await call.message.answer("Супер! Анкета и твоя запись отправлены админам. Обязательно проверь, чтобы у тебя были открыты личные сообщения, иначе мы не сможем тебе написать. Скоро свяжемся!")
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
    await message.answer("6. В какое время тебе удобнее всего заниматься? (Утро, день или вечер)")
    await state.set_state(StudentForm.time)

# Показываем превью анкеты Ученика
@router.message(StudentForm.time)
async def s_preview(message: Message, state: FSMContext):
    await state.update_data(time=message.text)
    data = await state.get_data()
    
    text = (
        f"Твоя анкета готова! Проверь, всё ли верно:\n\n"
        f"<b>Имя:</b> {data['name']}\n"
        f"<b>Возраст:</b> {data['age']}\n"
        f"<b>Уровень:</b> {data['level']}\n"
        f"<b>Цель:</b> {data['goal']}\n"
        f"<b>Часы в неделю:</b> {data['hours']}\n"
        f"<b>Удобное время:</b> {data['time']}"
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
    
    text = (
        f"📚 <b>Анкета Ученика | Global Talkers</b>\n\n"
        f"<b>Имя:</b> {data['name']}\n"
        f"<b>Ник:</b> {username}\n"
        f"<b>Возраст:</b> {data['age']}\n"
        f"<b>Уровень:</b> {data['level']}\n"
        f"<b>Цель:</b> {data['goal']}\n"
        f"<b>Часы в неделю:</b> {data['hours']}\n"
        f"<b>Удобное время:</b> {data['time']}"
    )
    
    await bot.send_message(chat_id=ADMIN_CHAT_ID, text=text, parse_mode="HTML")
    await call.message.answer("Готово! Анкета отправлена. Обязательно проверь, чтобы у тебя были открыты личные сообщения, иначе мы не сможем тебе написать. Скоро свяжемся!")
    await state.clear()

@router.callback_query(StudentForm.confirm, F.data == "s_edit")
async def s_edit_final(call: CallbackQuery, state: FSMContext):
    await call.message.edit_reply_markup(reply_markup=None)
    await call.message.answer("Без проблем, давай начнем сначала.\n\n1. Как к тебе обращаться?")
    await state.set_state(StudentForm.name)

async def handle_ping(request):
    return web.Response(text="Бот работает, не спать!")

# === ЗАПУСК ===
async def main():
    if not ADMIN_CHAT_ID:
        print("ВНИМАНИЕ: Не найден ADMIN_CHAT_ID! Проверь файл .env")
        return
        
    bot = Bot(token=TOKEN)
    print("Бот Global Talkers запущен...")
    app = web.Application()
    app.router.add_get('/', handle_ping)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()

    print("Обманка запущена, включаем бота...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
