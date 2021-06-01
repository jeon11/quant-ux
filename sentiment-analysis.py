# Imports the Google Cloud client library
import os
from google.cloud import language_v1


# set environment for credentials (need to be called with every start of instance)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/Jin/google-cloud-sdk/natural-language-api.json"

# Instantiates a client
client = language_v1.LanguageServiceClient()

# The text to analyze
text = u"This is just very good. Amazing representation of the artwork and stunning, but I see some flaws to it as well."
document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)

# Detects the sentiment of the text
sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment

print("Text: {}".format(text))
print("Sentiment: {}, {}".format(sentiment.score, sentiment.magnitude))


## References
# Setting up Natural Language API https://cloud.google.com/natural-language/docs/setup
# Installing Google SDK: https://cloud.google.com/sdk/docs/install
# Installing Natural Language API client library: https://cloud.google.com/natural-language/docs/reference/libraries
