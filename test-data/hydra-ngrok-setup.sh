#!/bin/sh

# run once to download ngrok

cd ~
curl https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz --output ngrok.tgz
tar xvzf ngrok.tgz -C ~/
ngrok config add-authtoken YOUR_AUTH_HERE
