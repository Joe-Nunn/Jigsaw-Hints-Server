#!/bin/sh
#SBATCH --cpus-per-task=8 --mem=0

# this script will run ngrok alongside any other commands...

# replace 5000 with port to use (flask default is 5000)
ngrok http 5000 > /dev/null &

# have to wait for ngrok to start
sleep 5

# get url (ngrok free tier is different every time)
echo URL is...
echo $(curl http://localhost:4040/api/tunnels | jq ".tunnels[0].public_url")

# enter commands after this for running the server...
# remember to activate the python virtual environment




echo Finished!
