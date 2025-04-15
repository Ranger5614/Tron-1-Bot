import time
import logging

class BotMonitor:
    def __init__(self, bot=None):
        self.bot = bot
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def monitor(self):
        while True:
            try:
                # Checking bot status, you can expand this with actual bot status checks
                self.logger.info("Checking bot status...")
                if self.bot:
                    status = self.bot.get_status()  # Assuming your bot has a method to get status
                    self.logger.info(f"Bot status: {status}")
                else:
                    self.logger.warning("No bot instance found.")
                time.sleep(60)  # Sleep for 1 minute between checks
            except Exception as e:
                self.logger.error(f"Error while monitoring bot: {str(e)}")
                time.sleep(60)  # Sleep for 1 minute before retrying in case of error