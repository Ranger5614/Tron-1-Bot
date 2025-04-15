from discord_webhook import DiscordWebhook, DiscordEmbed
import logging
import requests
import time
from datetime import datetime
import sys
import os
import json

# Add parent directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import config
try:
    import config
    # Use config for webhook if available
    WEBHOOK_URL = getattr(config, 'DISCORD_WEBHOOK_URL', 
                  "https://discordapp.com/api/webhooks/1360202061588987946/jfLROv1q5iURiLaauJuH5Y1e02H3XLhcZ_2TwFvQ-PlzxHUqmKDaHaN5dfO7AlqGCvxh")
    # Get other config values
    TRADING_PAIRS = getattr(config, 'TRADING_PAIRS', ["BTCUSDT", "ETHUSDT"])
    STRATEGY = getattr(config, 'STRATEGY', "TRON11")
    USE_TESTNET = getattr(config, 'USE_TESTNET', True)
except ImportError:
    # Fallback to hardcoded values
    WEBHOOK_URL = "https://discordapp.com/api/webhooks/1360202061588987946/jfLROv1q5iURiLaauJuH5Y1e02H3XLhcZ_2TwFvQ-PlzxHUqmKDaHaN5dfO7AlqGCvxh"
    TRADING_PAIRS = ["BTCUSDT", "ETHUSDT"]
    STRATEGY = "TRON11"
    USE_TESTNET = True

# Configure logging
logger = logging.getLogger("crypto_bot.notifier")

def send_trade_notification(pair, action, price, quantity, profit_loss=None, profit_loss_pct=None, message=None, conviction=None):
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
        conviction (float, optional): Signal conviction level (0.0-1.0).
    """
    try:
        # Use the webhook URL from config or fallback to hardcoded value
        webhook_url = WEBHOOK_URL
        
        if not webhook_url:
            logger.info("Discord webhook URL not configured. Skipping notification.")
            return True
            
        # Format price and quantity based on their types
        if isinstance(price, (int, float)):
            price_str = f"${price:.2f}"
        else:
            price_str = f"${price}"
            
        if isinstance(quantity, (int, float)):
            # Format quantity based on size
            if quantity < 0.001:
                quantity_str = f"{quantity:.8f}"
            elif quantity < 0.1:
                quantity_str = f"{quantity:.6f}"
            elif quantity < 1:
                quantity_str = f"{quantity:.4f}"
            else:
                quantity_str = f"{quantity:.3f}"
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
        
        payload = {}
        embeds = []
        
        if message:
            embed = {
                "title": "Cycle Update",
                "description": message,
                "color": int("808080", 16)
            }
            embeds.append(embed)
        else:
            # Add testnet indicator if in test mode
            testnet_indicator = "🧪 TESTNET " if USE_TESTNET else ""
            
            embed = {
                "title": f"{testnet_indicator}{action} {pair}",
                "description": f"Price: {price_str}\nQuantity: {quantity_str}",
                "color": int(color, 16),
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": f"{STRATEGY} Trading Bot"},
                "fields": []
            }

            # Calculate order value
            if isinstance(price, (int, float)) and isinstance(quantity, (int, float)):
                order_value = price * quantity
                embed["fields"].append({
                    "name": "Order Value",
                    "value": f"${order_value:.2f}",
                    "inline": True
                })

            if profit_loss is not None and profit_loss_pct is not None:
                # Format PnL for display
                if abs(profit_loss) < 0.01:
                    # Very small PnL, display more decimal places
                    formatted_pnl = f"${profit_loss:.6f}"
                else:
                    formatted_pnl = f"${profit_loss:.2f}"
                    
                embed["fields"].append({
                    "name": "Profit/Loss", 
                    "value": f"{formatted_pnl} ({profit_loss_pct:.2f}%)",
                    "inline": True
                })
                
            # Add conviction level if available (TRON 1.1 feature)
            if conviction is not None:
                # Format conviction as stars for visual representation
                if isinstance(conviction, (int, float)):
                    # Convert to scale of 1-5 stars
                    star_count = max(1, min(5, int(conviction * 5 + 0.5)))
                    stars = "⭐" * star_count
                    embed["fields"].append({
                        "name": "Signal Strength", 
                        "value": f"{stars} ({conviction:.2f})",
                        "inline": True
                    })
            
            embeds.append(embed)
        
        payload["embeds"] = embeds
            
        # Execute with retry logic for rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            response = requests.post(webhook_url, json=payload)
            
            if response.status_code == 429:  # Rate limited
                retry_after = response.json().get('retry_after', 5) / 1000.0
                logger.warning(f"Rate limited by Discord. Retrying after {retry_after} seconds.")
                time.sleep(retry_after + 0.5)  # Add a small buffer
            elif response.status_code == 204 or response.status_code == 200:  # Success
                # Both 204 and 200 are success codes for Discord webhooks
                if response.status_code == 200:
                    # If response has content, check if it has a message ID (success) or error message
                    try:
                        response_json = response.json()
                        if 'id' in response_json:
                            logger.info(f"Successfully sent {action} notification for {pair}")
                            return True
                        else:
                            logger.error(f"Discord error: {response_json}")
                            continue
                    except:
                        # If can't parse JSON, assume success for 200
                        logger.info(f"Successfully sent {action} notification for {pair}")
                        return True
                else:
                    # 204 is always success
                    logger.info(f"Successfully sent {action} notification for {pair}")
                    return True
            else:
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
        # Use the webhook URL from config or fallback to hardcoded value
        webhook_url = WEBHOOK_URL
        
        if not webhook_url:
            logger.info("Discord webhook URL not configured. Skipping notification.")
            return True
            
        # Create payload for direct API call instead of using DiscordWebhook
        payload = {"embeds": []}
        
        # Add testnet indicator if in test mode
        testnet_indicator = "🧪 TESTNET " if USE_TESTNET else ""
        
        # Create embed for status update
        embed = {
            "title": f"{testnet_indicator}{STRATEGY} Bot Status Update",
            "description": "Bot is active and monitoring the market",
            "color": int("808080", 16),  # Gray for status updates
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Trading Bot Status Update"},
            "fields": []
        }
        
        # Add current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        embed["fields"].append({
            "name": "Time", 
            "value": current_time, 
            "inline": False
        })
        
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
                
            # Add emoji indicators
            if signal == "BUY":
                emoji = "🟢"
            elif signal == "SELL":
                emoji = "🔴"
            elif signal == "HOLD":
                emoji = "⚪"
            elif signal == "ERROR":
                emoji = "⚠️"
            else:
                emoji = "❓"
                
            status_text += f"**{pair}**: {emoji} {signal} @ {price_str}\n"
            
            if signal != "HOLD":
                all_hold = False
        
        embed["fields"].append({
            "name": "Trading Pairs", 
            "value": status_text, 
            "inline": False
        })
        
        # Add overall status message
        if all_hold:
            embed["fields"].append({
                "name": "Status", 
                "value": "All trade indicators are on HOLD currently. Continuing to monitor the market.", 
                "inline": False
            })
        
        payload["embeds"].append(embed)
        
        # Send the webhook using direct API call
        response = requests.post(webhook_url, json=payload)
        
        # Both 204 and 200 are success codes for Discord webhooks
        if response.status_code == 204 or response.status_code == 200:
            logger.info(f"Successfully sent status update (status {response.status_code})")
            return True
        else:
            logger.error(f"Failed to send status update. Status: {response.status_code}, Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending status update: {str(e)}")
        return False

def send_cycle_update(trading_pairs, cycle_number=None):
    """
    Sends a notification for every trading cycle.
    
    Args:
        trading_pairs (list): List of trading pairs being monitored.
        cycle_number (int, optional): The current cycle number.
    """
    try:
        # Use the webhook URL from config or fallback to hardcoded value
        webhook_url = WEBHOOK_URL
        
        if not webhook_url:
            logger.info("Discord webhook URL not configured. Skipping notification.")
            return True
            
        # Create payload for direct API call
        payload = {"embeds": []}
        
        # Add testnet indicator if in test mode
        testnet_indicator = "🧪 TESTNET " if USE_TESTNET else ""
        
        # Create embed for cycle update
        embed = {
            "title": f"{testnet_indicator}Trading Cycle Update",
            "description": "🔄 Bot has completed a trading cycle 🔄",
            "color": int("5865f2", 16),  # Discord blue color
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "5-Minute Trading Cycle"},
            "fields": []
        }
        
        # Add current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        embed["fields"].append({
            "name": "Time", 
            "value": current_time, 
            "inline": True
        })
        
        # Add cycle number if provided
        if cycle_number is not None:
            embed["fields"].append({
                "name": "Cycle", 
                "value": f"#{cycle_number}", 
                "inline": True
            })
        
        # Add pairs being monitored
        pairs_text = ", ".join(trading_pairs)
        embed["fields"].append({
            "name": "Monitored Pairs", 
            "value": pairs_text, 
            "inline": False
        })
        
        payload["embeds"].append(embed)
        
        # Send the webhook using direct API call
        response = requests.post(webhook_url, json=payload)
        
        # Both 204 and 200 are success codes for Discord webhooks
        if response.status_code == 204 or response.status_code == 200:
            logger.info(f"Successfully sent cycle update (status {response.status_code})")
            return True
        else:
            logger.error(f"Failed to send cycle update. Status: {response.status_code}, Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending cycle update: {str(e)}")
        return False

def send_error_notification(error_message, details=None):
    """
    Sends an error notification to Discord.
    
    Args:
        error_message (str): The main error message
        details (dict, optional): Additional error details
    """
    try:
        # Use the webhook URL from config or fallback to hardcoded value
        webhook_url = WEBHOOK_URL
        
        if not webhook_url:
            logger.info("Discord webhook URL not configured. Skipping notification.")
            return True
            
        # Create payload for direct API call
        payload = {"embeds": []}
        
        # Create embed for error notification
        embed = {
            "title": "⚠️ Bot Error Detected ⚠️",
            "description": error_message,
            "color": int("ff0000", 16),  # Red for errors
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Trading Bot Error"},
            "fields": []
        }
        
        # Add current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        embed["fields"].append({
            "name": "Time", 
            "value": current_time, 
            "inline": True
        })
        
        # Add testnet status
        embed["fields"].append({
            "name": "Environment", 
            "value": "Testnet" if USE_TESTNET else "Live", 
            "inline": True
        })
        
        # Add details if provided
        if details:
            details_text = ""
            for key, value in details.items():
                details_text += f"**{key}**: {value}\n"
            
            embed["fields"].append({
                "name": "Details", 
                "value": details_text, 
                "inline": False
            })
        
        payload["embeds"].append(embed)
        
        # Send the webhook using direct API call
        response = requests.post(webhook_url, json=payload)
        
        # Both 204 and 200 are success codes for Discord webhooks
        if response.status_code == 204 or response.status_code == 200:
            logger.info(f"Successfully sent error notification")
            return True
        else:
            logger.error(f"Failed to send error notification. Status: {response.status_code}, Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending error notification: {str(e)}")
        return False
        
def test_webhook():
    """
    Test the Discord webhook to ensure it's working properly.
    """
    try:
        # Use the webhook URL from config or fallback to hardcoded value
        webhook_url = WEBHOOK_URL
        
        if not webhook_url:
            logger.info("Discord webhook URL not configured. Skipping test.")
            return False
            
        # Create simple payload for testing
        payload = {
            "content": f"Testing {STRATEGY} Bot webhook. If you see this message, your webhook is configured correctly!"
        }
        
        # Send test message
        response = requests.post(webhook_url, json=payload)
        
        # Both 204 and 200 are success codes for Discord webhooks
        if response.status_code == 204 or response.status_code == 200:
            logger.info(f"Webhook test successful! (status {response.status_code})")
            return True
        else:
            logger.error(f"Webhook test failed. Status: {response.status_code}, Response: {response.text}")
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
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Test Discord notifications')
    parser.add_argument('--test-error', action='store_true', help='Test error notification')
    parser.add_argument('--test-trade', action='store_true', help='Test trade notification')
    parser.add_argument('--test-cycle', action='store_true', help='Test cycle notification')
    parser.add_argument('--test-status', action='store_true', help='Test status notification')
    parser.add_argument('--test-all', action='store_true', help='Test all notification types')
    args = parser.parse_args()
    
    # Basic webhook test
    test_result = test_webhook()
    print(f"Webhook test {'successful' if test_result else 'failed'}")
    
    if args.test_all or args.test_error:
        # Test error notification
        error_test = send_error_notification("This is a test error message", {"test": True, "time": datetime.now().strftime("%H:%M:%S")})
        print(f"Error notification test {'successful' if error_test else 'failed'}")
    
    if args.test_all or args.test_trade:
        # Test trade notification
        trade_test = send_trade_notification("BTCUSDT", "BUY", 45000.50, 0.01, conviction=0.85)
        print(f"Trade notification test {'successful' if trade_test else 'failed'}")
    
    if args.test_all or args.test_cycle:
        # Test cycle update notification
        cycle_test = send_cycle_update(TRADING_PAIRS, 1)
        print(f"Cycle update test {'successful' if cycle_test else 'failed'}")
    
    if args.test_all or args.test_status:
        # Test status update notification
        signals = {pair: "HOLD" for pair in TRADING_PAIRS}
        signals[TRADING_PAIRS[0]] = "BUY" # Set first pair to BUY for testing
        prices = {pair: 50000.0 if "BTC" in pair else 3000.0 if "ETH" in pair else 100.0 for pair in TRADING_PAIRS}
        status_test = send_status_update(TRADING_PAIRS, signals, prices)
        print(f"Status update test {'successful' if status_test else 'failed'}")