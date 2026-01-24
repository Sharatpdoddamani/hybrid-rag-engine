"""
Discord bot integration for persona-based chat assistant
Run with: python discord_bot.py
"""
import discord
from discord.ext import commands
from chat_infer import chat_once
import asyncio
from collections import defaultdict
import os

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Store personas per channel
channel_personas = defaultdict(
    lambda: "You are a helpful, creative conversational assistant. Respond in character to create engaging conversations."
)

# Store conversation context (optional)
conversation_histories = defaultdict(list)

@bot.event
async def on_ready():
    print(f'✅ Bot logged in as {bot.user}')
    print(f'📊 Connected to {len(bot.guilds)} servers')
    print(f'👥 Serving {len(bot.users)} users')
    print('\n🎮 Commands:')
    print('  !chat <message> - Chat with the bot')
    print('  !setpersona <persona> - Set character persona for this channel')
    print('  !reset - Reset conversation history')
    print('  !help - Show this help message\n')

@bot.command(name='setpersona', help='Set the character persona for this channel')
async def set_persona(ctx, *, persona: str):
    """Set persona for current channel"""
    if len(persona) > 500:
        await ctx.send('❌ Persona too long! Keep it under 500 characters.')
        return
    
    channel_personas[ctx.channel.id] = persona
    conversation_histories[ctx.channel.id].clear()  # Clear history on persona change
    await ctx.send(f'✅ Persona set for this channel!\n```{persona}```')

@bot.command(name='chat', help='Chat with the persona-based assistant')
async def chat(ctx, *, message: str):
    """Chat with the bot"""
    if len(message) > 500:
        await ctx.send('❌ Message too long! Keep it under 500 characters.')
        return
    
    # Show typing indicator
    async with ctx.typing():
        persona = channel_personas[ctx.channel.id]
        
        # Add conversation context (last 3 exchanges)
        context = conversation_histories[ctx.channel.id][-6:]
        context_message = ""
        if context:
            context_message = "\n".join(context) + "\n"
        
        try:
            # Run in executor to not block
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                chat_once, 
                persona, 
                context_message + message
            )
            
            # Store in history
            conversation_histories[ctx.channel.id].append(f"User: {message}")
            conversation_histories[ctx.channel.id].append(f"Bot: {response}")
            
            # Keep only last 10 messages
            if len(conversation_histories[ctx.channel.id]) > 10:
                conversation_histories[ctx.channel.id] = conversation_histories[ctx.channel.id][-10:]
            
            await ctx.send(response)
        except Exception as e:
            await ctx.send(f'❌ Error: {str(e)}')

@bot.command(name='reset', help='Reset conversation history for this channel')
async def reset(ctx):
    """Reset conversation history"""
    conversation_histories[ctx.channel.id].clear()
    await ctx.send('✅ Conversation history reset!')

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        # Suggest using !chat command
        suggestion = f"💡 Tip: Use `!chat <your message>` to chat with the bot!\n"
        suggestion += f"Or use `!help` to see all commands."
        await ctx.send(suggestion)
    else:
        await ctx.send(f'❌ Error: {str(error)}')

def main():
    # Get token from environment variable
    token = os.getenv('DISCORD_BOT_TOKEN')
    
    if not token:
        print("❌ ERROR: DISCORD_BOT_TOKEN environment variable not set!")
        print("\n📝 How to get a token:")
        print("1. Go to https://discord.com/developers/applications")
        print("2. Create a new application")
        print("3. Go to 'Bot' section and create a bot")
        print("4. Copy the token")
        print("5. Run: set DISCORD_BOT_TOKEN=your_token_here")
        print("   Then: python discord_bot.py")
        return
    
    bot.run(token)

if __name__ == "__main__":
    main()

