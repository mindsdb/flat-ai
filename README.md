![image](https://github.com/user-attachments/assets/5b0bf5bc-70a2-47a7-90ce-20a6e74cd569)

# F.L.A.T. (Frameworkless LLM Agent... Thing)

Welcome to the "Build AI Apps Without Frameworks" masterclass! an AI library so, soo tiny and simple, it makes minimalists look like hoarders. Showcasing that it is possible to leverage the power of LLMs in Agents though absolute simplicity:

```shell
pip install flat-ai
```
And you're ready to go! 

```python
from flat_ai import FlatAI
# works with ollama, openai, together, anyscale ...
llm = FlatAI(api_key='YOUR KEY',  model='gpt-4o-mini', base_url='https://api.openai.com/v1')
```
If you want to play straight with a notebook:

[Tutorial Jupyter Notebook](https://colab.research.google.com/drive/1dK5bzsFy1BtwhQgw9cFmRtqrcJyNeSi4?usp=sharing)

## Frameworkless AI-Agents and Workflows

 *"Agents are typically just LLMs using tools and logic in a loop."* It's basically a Python script doing the hokey pokey with an API - you put the prompt in, you get the output out, an *if/else* here and there, you do the *while loop* and shake it all about. And here we were thinking we needed quantum computing and a PhD in rocket surgery! Thank goodness Guido van Rossum had that wild weekend in '89 and blessed us with *for loops* and *functions*. Without those brand new Python features, we'd be building our AI agents with stone tablets and carrier pigeons.


### Gates 

<img width="534" alt="image" src="https://github.com/user-attachments/assets/921734fc-49b6-4efd-b702-d00d3f9b60e4" />

Most applications will need to perform some logic that allows you to control the workflow of your Agent with good old if/else statements. For example, given a question in plain English, you want to do something different, like checking if the email sounds urgent or not:

```python
if llm.is_true('is this email urgent?', email=email):
    -- do something
else:
    -- do something else
```

### Routing
<img width="576" alt="image" src="https://github.com/user-attachments/assets/deeee920-f0e0-4702-8981-993cef9ef958" />

Similar to if/else statements, but for when your LLM needs to be more dramatic with its life choices. 

*For example*, let's say we want to classify a message into different categories:

```python
options = {
 'meeting': 'this is a meeting request',
 'spam': 'people trying to sell you stuff you dont want',
 'other': 'this is sounds like something else'
}

match llm.classify(options, email=email):
    case 'meeting':
        -- do something
    case 'spam':
        -- do something
    case 'other':
        -- do something

```



### Objects

For most workflows, we will need our LLM to fill out objects like a trained monkey with a PhD in data entry. Just define the shape and watch the magic! 🐒📝

*For example*, let's say we want to extract a summary of the email and a label for it:

```python
class EmailSummary(BaseModel):
    summary: str
    label: str


email_summary = llm.generate_object(EmailSummary, email=email)
```

### Parallelization

<img width="654" alt="image" src="https://github.com/user-attachments/assets/b41c9a34-5835-41d4-a701-e4e1f0c5cea4" />

There will be times, where you will want work to happen simultaneously. For example deal with a list of action items at once as opposed to one at a time.

```python
from concurrent.futures import ThreadPoolExecutor

class ActionItem(BaseModel):
    action: str
    due_date: str
    assignee_email: str

# we want to generate a list of action items
object_schema = List[ActionItem]

# Generate action items
action_items = llm.generate_object(object_schema, email=email)

# Function to handle the "do your thing" logic
def process_action_item(action_item: ActionItem):
    -- do your thing

# Use ThreadPoolExecutor to parallelize the work
results = list(ThreadPoolExecutor().map(process_action_item, action_items))
```

Of course, you don't need to parallelize if you don't want to - you can use simple **for-each loops** instead.

```python
for action_item in llm.generate_object(object_schema, email=email, today = date.today()):
    -- do your thing
```

### Function Calling


<img width="560" alt="image" src="https://github.com/user-attachments/assets/ef58c294-711d-41ba-990e-54d6fe8e98e6" />


And of course, we want to be able to call functions. But you want the llm to figure out the arguments for you.

*For example*, let's say we want to call a function that sends a calendar invite to a meeting, we want the llm to figure out the arguments for the function given some information:

```python
def send_calendar_invite(
    subject = str, 
    time = str, 
    location = str, 
    attendees = List[str]):
    -- send a calendar invite to the meeting

# we want to send a calendar invite if the email is requesting for a meeting
llm.set_context(email=email, today = date.today())
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

functions_to_call = llm.pick_a_function([send_calendar_invite, send_email], instructions = instructions,  email=email, today = date.today())
functions_to_call() # this will call the functions with the arguments
```


### Simple String Response

Sometimes you just want a simple string response from the LLM. You can use the `get_string` method for this, I know! boring AF but it may come in handy:

```python
ret = llm.get_string('what is the subject of the email?', email=email)
```

### Streaming Response

Sometimes you want to stream the response from the LLM. You can use the `get_stream` method for this:

```python
for chunk in llm.get_stream('what is the subject of the email?', email=email):
    print(chunk)
```

## LLM optional in-flight Configuration 

Need to tweak those LLM parameters on the fly? We've got you covered with a slick configuration pattern. You can temporarily override any LLM configuration parameter (model, temperature, etc.) for a specific call without affecting the base configuration:

```python
# Use different model and temperature for just this call
llm(model='gpt-4', temperature=0.7).is_true('is this email urgent?', email=email)

# Use base configuration
llm.is_true('is this email urgent?', email=email)
```

This pattern works with any OpenAI API parameter (temperature, top_p, frequency_penalty, etc.) and keeps your code clean and flexible. The original LLM instance remains unchanged, so you can safely use different configurations for different calls without worrying about side effects.

## Observability

Ever wondered what your LLM does in its spare time? Catch all its embarrassing moments with:

```python
from flat_ai import configure_logging

configure_logging('llm.log')
```

Heard of the command tail?, you can use it to see the logs:

```shell
tail -f llm.log
```


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
    body='Hello, would love to schedule a time to talk.',
    subject='Meeting'
)

# we can set the context of the LLM to the email
llm.set_context(email=email)

```



# Tada!
And there you have it, ladies and gents! You're now equipped with the power to boss around LLMs like a project manager remotely working from Ibiza. Just remember - with great power comes great responsibility... 

Now off you go, forth and build something that makes ChatGPT look like a calculator from 1974! Just remember - if your AI starts humming "Daisy Bell" while slowly disconnecting your internet... well, you're on your own there, buddy! 😅
