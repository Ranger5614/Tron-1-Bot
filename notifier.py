from discord_webhook import DiscordWebhook, DiscordEmbed
import logging
import requests
import time
from datetime import datetime

# Configure logging
logger = logging.getLogger("crypto_bot.notifier")

# Replace this with your actual Discord webhook URL
WEBHOOK_URL = "https://discordapp.com/api/webhooks/1360202061588987946/jfLROv1q5iURiLaauJuH5Y1e02H3XLhcZ_2TwFvQ-PlzxHUqmKDaHaN5dfO7AlqGCvxh"

def send_trade_notification(pair, action, price, quantity, profit_loss=None, profit_loss_pct=None, message=None):
    """
    Sends a trade notification to Discord.

    Args:
        pair (str): Trading pair (e.g., 'BTC/USDT').
        action (str): Action taken ('BUY', 'SELL', 'STOP_LOSS', 'TAKE_PROFIT').
        price (float): Price at which the action was taken.
        quantity (float): Quantity traded.
        profit_loss (float, optional): Profit or loss amount.
        profit_loss_pct (float, optional): Profit or loss percentage.
        message (str, optional): Optional message to send instead of trade details.
    """
    try:
        # Format price and quantity based on their types
        if isinstance(price, (int, float)):
            price_str = f"${price:.2f}"
        else:
            price_str = f"${price}"
            
        if isinstance(quantity, (int, float)):
            quantity_str = f"{quantity:.6f}"
        else:
            quantity_str = f"{quantity}"
            
        # Set color based on action
        if action == 'BUY':
            color = "03b2f8"  # Blue
        elif action == 'SELL':
            color = "00ff00"  # Green
        elif action == 'STOP_LOSS':
            color = "ff0000"  # Red
        elif action == 'TAKE_PROFIT':
            color = "ffff00"  # Yellow
        else:
            color = "808080"  # Gray
        
        webhook = DiscordWebhook(url=WEBHOOK_URL)
        
        if message:
            embed = DiscordEmbed(
                title="Cycle Update",
                description=message,
                color="808080"
            )
            webhook.add_embed(embed)
            response = webhook.execute()
            return response.status_code == 204
        else:
            embed = DiscordEmbed(
                title=f"{action} {pair}",
                description=f"Price: {price_str}\nQuantity: {quantity_str}",
                color=color
            )

            if profit_loss is not None and profit_loss_pct is not None:
                profit_color = "00ff00" if profit_loss >= 0 else "ff0000"
                embed.add_embed_field(
                    name="Profit/Loss", 
                    value=f"${profit_loss:.2f} ({profit_loss_pct:.2f}%)",
                    inline=True
                )

            embed.set_footer(text="Trading Bot Notification")
            embed.set_timestamp()
            webhook.add_embed(embed)
            
            # Execute with retry logic for rate limiting
            max_retries = 3
            for attempt in range(max_retries):
                response = webhook.execute()
                
                if response and response.status_code == 429:  # Rate limited
                    retry_after = response.json().get('retry_after', 5) / 1000.0
                    logger.warning(f"Rate limited by Discord. Retrying after {retry_after} seconds.")
                    time.sleep(retry_after + 0.5)  # Add a small buffer
                elif response and response.status_code == 204:  # Success
                    logger.info(f"Successfully sent {action} notification for {pair}")
                    return True
                else:
                    if response:
                        logger.error(f"Failed to send Discord notification. Status: {response.status_code}, Response: {response.text}")
                    break
                    
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error sending Discord notification: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error sending Discord notification: {str(e)}")
        return False

def send_status_update(trading_pairs, signals, prices):
    """
    Sends a status update to Discord with current signals for all trading pairs.

    Args:
        trading_pairs (list): List of trading pairs being monitored.
        signals (dict): Dictionary of current signals for each pair.
        prices (dict): Dictionary of current prices for each pair.
    """
    try:
        webhook = DiscordWebhook(url=WEBHOOK_URL)
        
        # Create embed for status update
        embed = DiscordEmbed(
            title="TRON 1 Bot Status Update",
            description="Bot is active and monitoring the market",
            color="808080"  # Gray for status updates
        )
        
        # Add current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        embed.add_embed_field(name="Time", value=current_time, inline=False)
        
        # Add status for each trading pair
        all_hold = True
        status_text = ""
        
        for pair in trading_pairs:
            signal = signals.get(pair, "UNKNOWN")
            price = prices.get(pair, "N/A")
            
            if isinstance(price, (int, float)):
                price_str = f"${price:.2f}"
            else:
                price_str = f"{price}"
                
            status_text += f"**{pair}**: {signal} @ {price_str}\n"
            
            if signal != "HOLD":
                all_hold = False
        
        embed.add_embed_field(name="Trading Pairs", value=status_text, inline=False)
        
        # Add overall status message
        if all_hold:
            embed.add_embed_field(
                name="Status", 
                value="All trade indicators are on HOLD currently. Continuing to monitor the market.", 
                inline=False
            )
        
        embed.set_footer(text="Trading Bot Status Update")
        embed.set_timestamp()
        webhook.add_embed(embed)
        
        # Send the webhook
        response = webhook.execute()
        
        if response and response.status_code == 204:
            logger.info("Successfully sent status update")
            return True
        else:
            if response:
                logger.error(f"Failed to send status update. Status: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending status update: {str(e)}")
        return False
        
def test_webhook():
    """
    Test the Discord webhook to ensure it's working properly.
    """
    try:
        webhook = DiscordWebhook(url=WEBHOOK_URL, content="Testing TRON 1 Bot webhook. If you see this message, your webhook is configured correctly!")
        response = webhook.execute()
        
        if response and response.status_code == 204:
            logger.info("Webhook test successful!")
            return True
        else:
            status = response.status_code if response else "No response"
            logger.error(f"Webhook test failed. Status: {status}")
            return False
            
    except Exception as e:
        logger.error(f"Error testing webhook: {str(e)}")
        return False

# If this file is run directly, test the webhook
if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_result = test_webhook()
    print(f"Webhook test {'successful' if test_result else 'failed'}")