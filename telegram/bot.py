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
TOKEN = '1763937310:AAHFmqPqQADl4Qimq3klBnrHm3WF1aDSHUs'
PHOTO = 1


def start(update: Update, _: CallbackContext) -> int:
    update.message.reply_text(
        'Добро пожаловать! Загрузите фотографию',
    )
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
        'Можете загрузить еще одну.',
    )
    return PHOTO


def cancel(update: Update, _: CallbackContext) -> int:
    user = update.message.from_user
    logger.info(f'User {user.first_name} canceled the conversation.')
    update.message.reply_text(
        'Спасибо, до встречи!', reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END


def main() -> None:
    # Create the Updater and pass it your bot's token.
    updater = Updater(TOKEN)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Add conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            PHOTO: [MessageHandler(Filters.photo, photo)],
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
