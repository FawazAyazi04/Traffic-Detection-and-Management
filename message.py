from twilio.rest import Client
import keys
import MODEL
client = Client(keys.account_sid, keys.auth_token)
message = client.messages.create(
    
        body = "Hello Muzammil Here",
        from_=keys.twilio_number,
        to=keys.target_number
)


print(message.body)