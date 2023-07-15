import base64
encoded_value = "LTkyMDExNjM1MjY4NDg4ODU5Mjk="
decoded_value = base64.b64decode(encoded_value).decode('utf-8')
print(decoded_value)
