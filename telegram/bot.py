import logging

import requests
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
)

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

URL_PARSE = 'http://web/parse/'
URL_ADD = 'http://web/add/'
TOKEN = '1763937310:AAHFmqPqQADl4Qimq3klBnrHm3WF1aDSHUs'
CHOOSING, PHOTO, ADD_PHOTO, ADD_FINISH = range(4)
_REPLY_KEYBOARD = [['Определить человека', 'Добавить человека', 'Выход']]


def start(update: Update, _: CallbackContext) -> int:
    update.message.reply_text(
        'Добро пожаловать! Выберите действие',
        reply_markup=ReplyKeyboardMarkup(_REPLY_KEYBOARD, one_time_keyboard=True),
    )
    return CHOOSING


def recognize_choice(update: Update, _: CallbackContext) -> int:
    update.message.reply_text(f'Загрузите фотографию')
    return PHOTO


def photo(update: Update, _: CallbackContext) -> int:
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    files = {'file': photo_file.download_as_bytearray()}
    response = requests.post(URL_PARSE, files=files)
    logger.info(response.text)
    logger.info(f'User {user.first_name} loaded photo')
    update.message.reply_text(
        f'Отлично. Вот результат: {response.text}',
    )
    # update.message.reply_photo(open('user_photo.jpg', 'rb'))
    update.message.reply_text(
        'Выберите дальнейшее действие',
        reply_markup=ReplyKeyboardMarkup(_REPLY_KEYBOARD, one_time_keyboard=True),
    )
    return CHOOSING


def add_choice(update: Update, _: CallbackContext) -> int:
    update.message.reply_text(f'Загрузите фотографию человека')
    return ADD_PHOTO


def add_photo(update: Update, context: CallbackContext) -> int:
    context.user_data['photo_file'] = update.message.photo[-1].get_file()
    update.message.reply_text(f'Введите имя человека')
    return ADD_FINISH


def add_finish(update: Update, context: CallbackContext) -> int:
    name = update.message.text
    files = {'file': context.user_data['photo_file'].download_as_bytearray()}
    requests.post(URL_ADD, files=files, data={'name': name})
    update.message.reply_text(
        'Фотография загружена. Выберите дальнейшее действие',
        reply_markup=ReplyKeyboardMarkup(_REPLY_KEYBOARD, one_time_keyboard=True),
    )
    return CHOOSING


def cancel(update: Update, _: CallbackContext) -> int:
    user = update.message.from_user
    logger.info(f'User {user.first_name} canceled the conversation.')
    update.message.reply_text(
        'Спасибо, до встречи!', reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END


def test_photo(update: Update, _: CallbackContext) -> int:
    photo_file = update.message.photo[-1].get_file()
    files = {'file': photo_file.download_as_bytearray()}
    response = requests.post('http://web/test_normalize/', files=files)
    if response.status_code == 200:
        update.message.reply_photo(response.content)
    return CHOOSING


def main() -> None:
    # Create the Updater and pass it your bot's token.
    updater = Updater(TOKEN)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Add conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            CHOOSING: [
                MessageHandler(Filters.regex('^Определить человека$'), recognize_choice),
                MessageHandler(Filters.regex('^Добавить человека$'), add_choice),
                MessageHandler(Filters.regex('^Выход$'), cancel),
                MessageHandler(Filters.photo, test_photo),
            ],
            PHOTO: [MessageHandler(Filters.photo, photo)],
            ADD_PHOTO: [MessageHandler(Filters.photo, add_photo)],
            ADD_FINISH: [MessageHandler(Filters.text, add_finish)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    dispatcher.add_handler(conv_handler)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
