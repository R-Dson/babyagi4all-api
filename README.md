# BabyAGI4All Using Oobabooga's Web Interface And Simple Online Searching 

A small autonomous AI agent based on [BabyAGI](https://github.com/yoheinakajima/babyagi) by Yohei Nakajima.
</br>

Originally made to work with GPT4ALL on CPU by kroll-software [here](https://github.com/kroll-software/babyagi4all). This is a small fork of a small fork to make it compatible with the API from [oobabooga's web interface](https://github.com/oobabooga/text-generation-webui) and some online searching.
</br>

100% open source, 100% local, no API-keys needed.
</br>

# Installation:
Prerequisite: Install https://github.com/oobabooga/text-generation-webui.

1. Clone this repository
2. Install the requirements: *pip install -r requirements.txt*
3. Copy the file .env.example to .env, and edit it to reflect your host name, port, and desired goal for the agent. There are other parameters as well.

Now, start up oobabooga's ui with the --api option. It should be listening on host "localhost" and port "5000" by default, meaning if you haven't changed anything about the config, you probably don't have to change those fields in .env. You also have to load a model; you can do this by just going to the web ui normally, or by passing the --model argument to server.py.

Then run *python babyagi.py*
</br>

Seriously all credits to kroll-software, all I did was plug it into Ooba's api :)
