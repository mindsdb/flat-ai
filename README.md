
![image](https://github.com/user-attachments/assets/6521d673-3b62-4b60-a4c9-26b34e6e1c31)

# F.L.A.T. (FlawLess Agents... Thing)

Building AI-Agents should be pretty simple, *"they are typically just LLMs calling functions and logic in a loop."* 

However, often times watching AI-Agents try to complete tasks is like watching a drunk person trying to solve a Rubik's cube. Entertaining? Yes. Reliable? Not always!
 
Instead of asking an LLM to "do the whole thing" (which is indeed prone to inconsistency), a FLAT approach puts control and predictability to LLM interactions by treating them more like traditional programming constructs but enhanced with LLM's natural language understanding. like:

- Binary decisions [(gates)](#gates)
- Classification [(match/case routing)](#routing)
- Structured data extraction [(using Pydantic models)](#objects)
- Tool calling [(Well typed function calls)](#function-calling)
- Parallelization [(Multithreading)](#parallelization)
- Observability [(Well formatted logs)](#observability)
  
Welcome F.L.A.T, an AI library so, soo tiny and simple, it makes minimalists look like hoarders. 

[Tutorial on Google Colab Notebook](https://colab.research.google.com/drive/1dK5bzsFy1BtwhQgw9cFmRtqrcJyNeSi4?usp=sharing)

## Setup 
```shell
pip install flat-ai
```
And you're ready to go! 

```python
from flat_ai import FlatAI
# works with ollama, openai, together, anyscale ... (just pass base_url=api_url)
llm = FlatAI(api_key='YOUR KEY',  model='gpt-4o-mini')
```

## Minimalistic AI-Agents with Python constructs
Thank goodness [Guido van Rossum](https://en.wikipedia.org/wiki/Guido_van_Rossum) had that wild pizza-night in '89 and blessed us with if/else, *for loops* and *functions*. Without those brand new Python features, we'd be building our AI agents with stone tablets and carrier pigeons. And here we were thinking we needed quantum computing and a PhD in rocket surgery! Let's showcase that it is possible to build Agents though absolute simplicity:


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

*For example*, let's deal with a list of action items at once as opposed to one at a time.

```python

class ActionItem(BaseModel):
    action: str
    due_date: str
    assignee_email: str

# we want to generate a list of action items
object_schema = List[ActionItem]

# deal with each action item
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
ret = llm.call_function(send_calendar_invite, email=email, today = date.today())
``` 

### Parallelization


<img width="654" alt="image" src="https://github.com/user-attachments/assets/b41c9a34-5835-41d4-a701-e4e1f0c5cea4" />


Sometimes you want to pick functions from a list of functions. And then call them all in parallel.

*For example*, let's say you want to send emails and calendar invites from a list of action items discussed in an email:

```python
def send_calendar_invites(subject = str, time = str, location = str, attendees = List[str]):
    -- send a calendar invite to the meeting

def send_email(name = str, email_address_list = List[str], subject = str, body = str):
    -- send an email

instructions = """
extract list of action items and call the funcitons required
"""

functions_to_call = llm.get_functions([send_calendar_invite, send_email], instructions = instructions, email=email, current_date = date.today())
# picks the functions that should be called and returns a parallel callable object, where each function is called in a separate thread.
results = functions_to_call()
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


## Painless Global Context

Sometimes it get's a bit annoying to pass the context every single time, so we're making it brain-dead simple with these methods to pass the context when we need it, and then clear it when we don't:


So we're making it brain-dead simple with these methods to pass the context when we need it, and then clear it when we don't:
- `set_context`: Dump any object into the LLM's memory banks
- `add_context`: Stack more stuff on top, like a context burrito
- `clear_context`: For when you want the LLM to forget everything, like the last 10 minutes of your life ;)
- `delete_from_context`: Surgical removal of specific memories

So lets say for example you want to set context that you want to avoid having to pass every single time

```python

# we can set global context, that will always be passed on every call, unless you want to remove it (which you can at any point in time)
llm.set_context(current_date=today(), user_name="Bob McPlumber", age="22", ....)
-- do some stuff
# oops he is actually 29, lets update that, and also, now it's fav color is pink
llm.add_context(age=29, fav_color="pink")
-- do some other stuff
# well turns out pink was not it, user cant make it's mind, no sweat
llm.delete_from_context(fav_color)
-- do something else
# no more global context is needed
llm.clear_context()
```



# Tada!
And there you have it, ladies and gents! You're now equipped with the power to boss around LLMs like a project manager remotely working from Ibiza. Just remember - with great power comes great responsibility... 

Now off you go, forth and build something that makes ChatGPT look like a calculator from 1974! Just remember - if your AI starts humming "Daisy Bell" while slowly disconnecting your internet... well, you're on your own there, buddy! 😅
