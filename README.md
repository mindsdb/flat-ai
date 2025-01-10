
# F.L.A.T. (Frameworkless LLM Agent... Thing)

Welcome to the "Build AI Apps Without Frameworks" masterclass! - 
Inspired on Anthropic's scholarly tome about [**building effective agents**.](https://www.anthropic.com/research/building-effective-agents) Too busy to read their post? Here's a spanky video summary by the legend Matt Berman ([here](https://www.youtube.com/watch?v=0v7TQIh_kes)).

Anywho, want to try this lib out?

```shell
pip install flat-ai
```

Or clone the repo, or DIY, this is an AI library so, soo tiny, it makes minimalists look like hoarders, and it should be be able to code one yourself... 

> **Oh wait I did not even code it myself...**. I simply asked an LLM to write it for me based on this very same README.md file. And it did. And it works. So then I asked it to solve all problems to life, the universe and everything, and here I am, not blinking, looking at the number 42... *We might have just entered the age where some dairy products last longer than agentic frameworks.*

## Let's get started

Best and fastest way, is to get your teeth to it:

[Tutorial Jupyter Notebook](/tutorial.ipynb)

# Features

It's basically LLM and Clean Python building blocks: We're talking IF/ELSE statements that actually get sarcasm, Loops made out of thin air, Switch cases with attitude, and Functions that don't need a GPS to find their purpose. It's regular Python syntax meets AI wizardry - what could possibly go wrong? 🧙‍♂️

## LLM Logic Blocks

Think Python code, but with an LLM brain transplant! 


### IF/ELSE Statements

Most applications will need to perform some logic that allows you to control the workflow of your Agente with good old if/else statements. For example, given a question in plain English, you want to do something different, like checking if the email sounds urgent or not:

```python

llm.set_context(email=email)

if llm.true_or_false('is this email urgent?'):
    -- do something
else:
    -- do something else

llm.clear_context()
```


### Switch case

Similar to if/else statements, but for when your LLM needs to be more dramatic with its life choices. 

*For example*, let's say we want to classify a message into different categories:

``` python

options = {
 'meeting': 'this is a meeting request',
 'spam': 'people trying to sell you stuff you dont want',
 'other': 'this is sounds like something else'
 }

llm.set_context(email=email)

match llm.get_key(options):
    case 'meeting':
        # you can add more context whenever you want
        llm.add_context(meeting=True)
    case 'spam':
        llm.add_context(spam=True)
    case 'other':
        -- do something


```

### Objects

Need your LLM to fill out objects like a trained monkey with a PhD in data entry? Just define the shape and watch the magic! 🐒📝

*For example*, let's say we want to extract a summary of the email and a label for it:

```python
class EmailSummary(BaseModel):
    summary: str
    label: str


llm.set_context(email=email)

ret = llm.generate_object(EmailSummary)

```


### Loops

Loops: Because all programming languages have them, and making your LLM do repetitive tasks is like having a genius do your laundry - hilarious but effective! Want a list of things? Just throw a schema at it and watch it spin like a hamster on a crack coated wheel. 

*For example*, let's say we want to extract a list of action items from an email

```python
class ActionItem(BaseModel):
    action: str
    status: str
    priority: str
    due_date: str
    assignee_name: str
    assignee_email: str

object_schema = List[ActionItem]

llm.set_context(email=email)

if llm.true_or_false('are there action items in this email?'):
    for action_item in llm.generate_object(object_schema):
        -- do something
```

### Function Calling

And of course, we want to be able to call functions. But you want the llm to figure out the arguments for you.

*For example*, let's say we want to call a function that sends a calendar invite to a meeting, we want the llm to figure out the arguments for the function given some information:
```python


def send_calendar_invite(
    subject = str, 
    time = str, 
    location = str, 
    attendees = List[str]):
    -- send a calendar invite to the meeting

llm.set_context(email=email)

if llm.true_or_false('is this an email requesting for a meeting?'):
    ret = llm.call_function(send_calendar_invite)

``` 

### Function picking

Sometimes you want to pick a function from a list of functions. You can do that by specifying the list of functions and then having the LLM pick one.

*For example*, let's say we want to pick a function from a list of functions:

```python

def send_calendar_invites(
    subject = str, 
    time = str, 
    location = str, 
    attendees = List[str]):
    -- send a calendar invite to the meeting

def send_email(
    name = str,
    email_address_list = List[str],
    subject = str,
    body = str):
    -- send an email

instructions = """
You are a helpful assistant that can send emails and schedule meetings.
You can pick a function from the list of functions and then call it with the arguments you want.
if:
    the email thread does not contain details about when people are available, please send an email to the list of email addresses, requesting for available times.
else
    send a calendar invites to the meeting
"""

llm.set_context(email=email)

# pick the function and the arguments
function, args = llm.pick_a_function(instructions, [send_calendar_invite, send_email])

# call the function with the arguments
function(**args)
```


### Simple String Response

Sometimes you just want a simple string response from the LLM. You can use the `get_string` method for this, I know! boring AF but it may come in handy:

```python

llm.set_context(email=email)

ret = llm.get_string('what is the subject of the email?')

```

### Streaming Response

Sometimes you want to stream the response from the LLM. You can use the `get_stream` method for this:

```python

llm.set_context(email=email)

for chunk in llm.get_stream('what is the subject of the email?'):
    print(chunk)

```


## OpenAI API based

As you are just about to see, this tiny library is designed to talk to LLMs that are served through an OpenAI API compatible endpoint (as they all should). 
But, scare yourself not when you read the words OpenAI. Because, you will still be able to play with all kinds of models and providers - OpenAI or not - using the same API ([Ollama](https://ollama.com/blog/openai-compatibility) as seen below, [Groq](https://console.groq.com/docs/openai), etc). Because, Most of them have OpenAI API compatible endpoints.

```python
import openai
from flat_ai import FlatAI

# Create client
client = openai.OpenAI(
    # base_url = 'http://localhost:11434/v1', -- if you want ollama
    api_key=<your api key>, # required, but unused
)

llm = FlatAI(client=client, model='gpt-4o-mini-2024-07-18')
```

With that, and two more simple simple steps, you are ready to start building your own AI agents.

## Painless Context

Ever tried talking to an LLM? You gotta give it a "prompt" - fancy word for "given some context {context}, please do something with this text, oh mighty AI overlord." But here's the optimization: constantly writing the code to pass the context to an LLM is like telling your grandparents how to use a smartphone... every. single. day. 

So we're making it brain-dead simple with these methods to pass the context when we need it, and then clear it when we don't:
- `set_context`: Dump any object into the LLM's memory banks
- `add_context`: Stack more stuff on top, like a context burrito
- `clear_context`: For when you want the LLM to forget everything, like the last 10 minutes of your life ;)
- `delete_from_context`: Surgical removal of specific memories

So lets say for example we want our LLM to start working magic with an email. You add the email to the context:

```python
from pydantic import BaseModel

# for the following examples, we will be using the following object
class Email(BaseModel):
    to_email: str
    from_email: str
    body: str
    subject: str

email = Email(
    to_email='john@doe.com',
    from_email='jane@doe.com',
    body='Hello, would love to shedule a time to talk.',
    subject='Meeting'
)

# we can set the context of the LLM to the email
llm.set_context(email=email)

```



# Tada!
And there you have it, ladies and gents! You're now equipped with the power to boss around LLMs like a project manager remotely working from Ibiza. Just remember - with great power comes great responsibility... and the occasional hallucination where your AI assistant thinks it's a pirate-ninja-astronaut.

Now off you go, forth and build something that makes ChatGPT look like a calculator from 1974! Just remember - if your AI starts humming "Daisy Bell" while slowly disconnecting your internet... well, you're on your own there, buddy! 😅
